import os
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

class ResearchState(TypedDict):
    """State for the research assistant workflow."""
    query: str
    research_plan: Optional[List[str]]
    current_step: Optional[int]
    retrieved_information: Optional[List[Dict[str, Any]]]
    analysis: Optional[str]
    summary: Optional[str]
    follow_up_questions: Optional[List[str]]
    final_report: Optional[str]
    messages: List[Dict[str, Any]]
    errors: Optional[List[str]]
    status: str 

_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

def set_llm(new_llm: ChatOpenAI) -> None:
    global _llm
    _llm = new_llm

def create_research_plan(state: ResearchState) -> ResearchState:
    """Create a reaearch plan based on the user's query."""
    query = state["query"]

    system_msg = SystemMessage(content="""
You are a research a planning assistant. Your job is to break down a research query into specific, actionable steps, Create a structured research plan with 3-5 clear steps.""")
    human_msg = HumanMessage(content=f"Create a research plan for the following query: {query}")
    response = _llm.invoke([system_msg, human_msg])
    research_plan = response.content.split("\n")
    research_plan = [step for step in research_plan if step.strip()]

    new_state = state.copy()
    new_state["research_plan"] = research_plan
    new_state["current_step"] = 0
    new_state["status"] = "planning"
    new_state["messages"].append({"role": "assistant", "content": f"Research plan created:\n" + "\n".join(research_plan)})

    return new_state

def execute_research_step(state: ResearchState) -> ResearchState:
    """Execute the current research step and retrieve information."""
   
    new_state = state.copy()

    if not state.get("research_plan") or state.get("current_step") is None:
        new_state["errors"] = new_state.get("errors", []) + ["No research plan or current step defined."]
        new_state["status"] = "error"
        return new_state
    
    current_step = state["current_step"]
    research_plan = state["research_plan"]

    if current_step >= len(research_plan):
        new_state["status"] = "analyzing"
        return new_state
    
    step_description = research_plan[current_step]

    # Create messages for step execution
    system_msg = SystemMessage(content="""Convert the research step into specific search queries that would help gather information for this step. Provide 2-3 search queries.""")
    human_msg = HumanMessage(content=f"Research step: {step_description}")
    response = _llm.invoke([system_msg, human_msg])
    search_queries = response.content.split("\n")
    search_queries = [q.strip() for q in search_queries if q.strip()]
    retrieved_information = []

    for query in search_queries:
        try: 
            search_result = {
                "query": query,
                "results": f"Simulated search results for query: {query}"
            }
            retrieved_information.append(search_result)
        except Exception as e:
            new_state["errors"] = new_state.get("errors", []) + [f"Error retrieving information for query '{query}': {str(e)}"]
    
    if "retrieved_information" not in new_state or new_state["retrieved_information"] is None:
        new_state["retrieved_information"] = []

    new_state["retrieved_information"].extend(retrieved_information)
    new_state["current_step"] += 1
    new_state["status"] = "researching"
    new_state["messages"].append({"role": "assistant", "content": f"Completed research step {current_step + 1}: {step_description}"})
    return new_state


def analyze_information(state: ResearchState) -> ResearchState:
    """Analyze the retrieved information and extract key insights."""
    new_state = state.copy()

    if not state.get("retrieved_information"):
        new_state["errors"] = new_state.get("errors", []) + ["No retrieved information to analyze."]
        new_state["status"] = "error"
        return new_state
    
    all_information =""
    for info in state["retrieved_information"]:
        all_information += f"Query: {info['query']}\nResults: {info['results']}\n\n"

    system_msg = SystemMessage(content="""
     Analyze the following research information. Identify key insights, patterns, and potential conclusions. Be objective and through.""")
    human_msg = HumanMessage(content=f"Information to analyze:\n{all_information}")
    response = _llm.invoke([system_msg, human_msg])

    new_state["analysis"] = response.content
    new_state["status"] = "summarizing"
    new_state["messages"].append({"role": "assistant", "content": "Information analyzed and key insights extracted."})
    return new_state


def generate_summary(state: ResearchState) -> ResearchState:
    """Generate a concise summary of the research findings."""
    new_state = state.copy()

    if not state.get("analysis"):
        new_state["errors"] = new_state.get("errors", []) + ["No analysis available to summarize."]
        new_state["status"] = "error"
        return new_state
    
    system_msg = SystemMessage(content="""
    Create a comprehensive yet concise summary of the research findings based on the following analysis. Highlight the most important insights and conclusions.""")
    human_msg = HumanMessage(content=f"Analysis to summarize:\n{state['analysis']}")
    response = _llm.invoke([system_msg, human_msg])

    question_system_msg = SystemMessage(content="""Based on the research analysis,, suggest 3-5 follow-up questions that could deepen understanding or explore related areas.""")
    question_human_msg = HumanMessage(content=f"Analysis to generate follow-up questions:\n{state['analysis']}")
    question_response = _llm.invoke([question_system_msg, question_human_msg])

    follow_up_questions = question_response.content.split("\n")
    follow_up_questions = [q.strip() for q in follow_up_questions if q.strip()]

    new_state["summary"] = response.content
    new_state["follow_up_questions"] = follow_up_questions
    new_state["status"] = "completed"
    new_state["messages"].append({"role": "assistant", "content": f"Research complete. Summary:\n\n{response.content}\n\nFollow-up questions:\n" + "\n".join(follow_up_questions)})
    return new_state


def should_continue_research(state: ResearchState) -> str:
    """Determine if we shoiuld continue with the research process"""

    if state.get("status") == "error":
        return "generate_summary"
    if state.get("status") == "planning":
        return "execute_research"
    if state.get("status") == "researching":
        if state.get("current_step") < len(state.get("research_plan", [])):
            return "execute_research"
        else:
            return "analyze_information"

    if state.get("status") == "analyzing":
        return "generate_summary"
    if state.get("status") == "summarizing" or state.get("status") == "completed":
        return "end"

    return "end"    


def build_research_graph() :
    """Build the research assistant workflow graph."""
    # Create a new graph
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("create_research_plan", create_research_plan)
    graph.add_node("execute_research", execute_research_step)
    graph.add_node("analyze_information", analyze_information)
    graph.add_node("generate_summary", generate_summary)

    # Add edges
    graph.add_conditional_edges(
        "create_research_plan",
        should_continue_research,
        {
            "execute_research": "execute_research",
            "generate_summary": "generate_summary",
            "end": END
        }
    )

    graph.add_conditional_edges(
        "execute_research",
        should_continue_research,
        {
            "execute_research": "execute_research",
            "analyze_information": "analyze_information",
            "generate_summary": "generate_summary",
            "end": END
        }
    )

    graph.add_conditional_edges(
        "analyze_information",
        should_continue_research,
        {
            "generate_summary": "generate_summary",
            "end": END
        }
    )

    graph.add_conditional_edges(
        "generate_summary",
        should_continue_research,
        {
            "end": END
        }   
    )

    # Set the entry point
    graph.set_entry_point("create_research_plan")

    # Compile the graph
    return graph.compile()

  


class ResearchAssistantLangGraph:
    """Research Assistant using LangGraph for workflow orchestration."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the Research Assistant with LangGraph workflow."""
        if llm is not None:
            set_llm(llm)
        self.graph = build_research_graph()

    def execute_research(self,query: str) -> Dict[str,Any]:

        # Initialize the state 
        initial_state = {
            "query": query,
            "research_plan": None,
            "current_step": None,
            "retrieved_information": None,
            "analysis": None,
            "summary": None,
            "follow_up_questions": None,
            "final_report": None,
            "messages": [{"role": "user", "content": query}],
            "errors": [],
            "status": "planning"
        }

        # Execute the graph
        result = self.graph.invoke(initial_state)
        return result
