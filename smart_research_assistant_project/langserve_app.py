import os
from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, Field
from smart_research_assistant.langchain_module import ResearchAssistantLangchain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph
from smart_research_assistant.langsmith_module import ResearchAssistantLangSmith

load_dotenv()

app = FastAPI(
    title="Smart Research Assistant API",
    description="API for the Smart Research Assistant using LangServe",
    version="1.0"
)
langchain_module = ResearchAssistantLangchain()
langgraph_module = ResearchAssistantLangGraph()
langsmith_module = ResearchAssistantLangSmith()


class ResearchInput(BaseModel):
    query: str = Field(..., description="The research query or topic to investigate.")
    enable_tracing: bool = Field(False, description="Whether to enable LangSmith tracing for this research execution.")

class ResearchOutput(BaseModel):
    summary: str = Field(..., description="Summary of the research topic or query.")
    research_plan: List[str] = Field(default=[], description="List of research steps to execute.")
    follow_up_questions: List[str] = Field(default=[], description="List of follow-up questions to ask after executing research steps.")
    analysis: str = Field("", description="Analysis of the retrieved information.")


def research_function(input_data: Dict[str,Any],config: RunnableConfig = None) -> Dict[str, Any]:
    try:
        query = input_data.get("query", "")
        if not query:
            return {
                "summary": "No query provided.",
                "research_plan": [],
                "follow_up_questions": [],
                "analysis": ""
            }

        enable_tracing = input_data.get("enable_tracing", False)

        try:
            if enable_tracing:
                result = langsmith_module.execute_with_tracing(query)
                if "result" in result:
                    data = result["result"]
                else:
                    data = result
            else:
                data = langgraph_module.execute_research(query)

            return {
                "summary": data.get("summary", ""),
                "research_plan": data.get("research_plan", []),
                "follow_up_questions": data.get("follow_up_questions", []),
                "analysis": data.get("analysis", "")
            }
        except Exception as e:
            print(f"Error executing research function: {e}")
            return {
                "summary": "",
                "research_plan": [],
                "follow_up_questions": [],
                "analysis": f"Error executing research: {e}"
            }

    except Exception as e:
        print(f"Error executing research function: {e}")
        return {
            "summary": "",
            "research_plan": [],
            "follow_up_questions": [],
            "analysis": f"Error executing research: {e}"
        }
    
research_chain = RunnableLambda(research_function)    

class SummaryInput(BaseModel):
    text: str = Field(..., description="Text to summarize.")

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="Generated summary")


def summarize_function(input_data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
    try:
        text = input_data.get("text", "")
        if not text:
            return {"summary": "No text provided for summarization."}
        
        summary = langchain_module.summarize_document(text)
        return {"summary": summary}
    except Exception as e:
        print(f"Error executing summarize function: {e}")
        return {"summary": f"Error during summarization: {e}"}

summarize_chain = RunnableLambda(summarize_function)

add_routes(
    app,
    research_chain,
    path="/research",
    input_type=ResearchInput,
    output_type=ResearchOutput
)
add_routes(
    app,
    summarize_chain,
    path="/summarize",
    input_type=SummaryInput,
    output_type=SummaryOutput
)

def echo_function(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": f"Received: {input_data}"}
echo_chain = RunnableLambda(echo_function)

class EchoInput(BaseModel): 
    text: str = Field(..., description="Text to echo .")

class EchoOutput(BaseModel):
    message: str = Field(..., description="Echo message.")    

add_routes(
    app,
    echo_chain,
    path="/echo",
    input_type=EchoInput,
    output_type=EchoOutput
)    

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Smart Research Assistant API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
  
