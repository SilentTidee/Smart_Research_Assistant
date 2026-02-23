import os 
import uuid 
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tracers import LangChainTracer

from smart_research_assistant.langchain_module import ResearchAssistantLangchain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph

load_dotenv()

class MockLangSmithClient:

    def __init__(self,*args,**kwargs):
        pass

    def create_dataset(self, *args, **kwargs):
        return "mock-dataset"
    
    def create_example(self, *args, **kwargs):
        return "mock-example"
    
    def create_feedback(self, *args, **kwargs):
        return "mock-feedback"
    
    def list_runs(self, *args, **kwargs):
        return []
    
try:
    from langsmith import Client as LangSmithClient
    langsmith_client = LangSmithClient(
        api_key=os.getenv("LANGCHAIN_API_KEY"),
        api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )    
except Exception as e:
    print(f"Warning: LangSmith client initialization failed: {e}")
    print("Using MockLangSmithClient instead.")
    langsmith_client = MockLangSmithClient()




class ResearchAssistantLangSmith:
    
    def __init__(self):
        """Initialize the Research Assistant with LangSmith tracing."""
        try:
            self.project_name = os.getenv("LANGCHAIN_PROJECT", "smart-research-assistant")
            self.tracer = LangChainTracer(project_name=self.project_name)
            self.langchain_module = ResearchAssistantLangchain()
            self.traced_llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                callbacks=[self.tracer]
            )
            self.langchain_module.llm = self.traced_llm
            self.langgraph_module = ResearchAssistantLangGraph(llm=self.traced_llm)
            self.tracing_enabled = True
        except Exception as e:
            print(f"Warning: LangSmith tracing disabled: {e}")
            self.langchain_module = ResearchAssistantLangchain()
            self.langgraph_module = ResearchAssistantLangGraph()
            self.tracing_enabled = False
            self.project_name = os.getenv("LANGCHAIN_PROJECT", "smart-research-assistant")


    def execute_with_tracing(self,query: str) -> Dict[str, Any]:

        if not self.tracing_enabled:
            result = self.langgraph_module.execute_research(query)
            return {"result": result}

        run_id = str(uuid.uuid4())

        try:
            result = self.langgraph_module.execute_research(query)
            return{
                "result": result,
                "run_id": run_id,
                "project": self.project_name
            }   
        except Exception as e:
            print(f"Error during execution with tracing: {e}")
            result = self.langgraph_module.execute_research(query)
            return {"result": result}

    def log_feedback(self, run_id: str, feedback_type: str, score: float, comment: Optional[str] = None):
       """
       Log feedback for a specific run
       
       Args:
              run_id:ID of the run provide feedback for 
              feedback_type: Type of feedback (e.g., "quality", "relevance")
              score: Score from 1-10
                comment: Optional comment providing additional context for the feedback
       """
       if not self.tracing_enabled:
           print(f"Feedback logged (mock): {feedback_type} - {score} - {comment}")
           return
       
       try:
           langsmith_client.create_feedback(
               run_id=run_id,
               key = feedback_type,
               score = score,
               comment = comment
            )
           print(f"Feedback logged: {feedback_type} - {score} - {comment}")
       except Exception as e:
            print(f"Error logging feedback: {e}")



    def analyze_performance(self,project_name: str = "smart_research_assistant") -> Dict[str, Any]:
        """
        Analyze performance of the research assistant
        
        Args:
            project_name: Name of the project to analyze

        Returns:
            Performance metrices
        """
        if not self.tracing_enabled:
            return{
                "message": "LangSmith tracing is disabled.Performance metrics not available",
                "total_runs": 0,
                "error_rate": 0,
                "avg_latency_seconds": 0,
                "project_name": project_name
            }            
        try:
            runs = langsmith_client.list_runs(project_name=project_name,execution_order = 1)
            total_runs = 0
            error_runs = 0
            avg_latency =0

            for run in runs:
                total_runs += 1
                if getattr(run, "error", None):
                    error_runs += 1

                start_time = getattr(run, "start_time", None)
                end_time = getattr(run, "end_time", None)

                if start_time and end_time:
                    try:
                        if isinstance(start_time, str) and isinstance(end_time, str):
                           from datetime import datetime
                           start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                           end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))    
                           avg_latency += (end - start).total_seconds()
                        else:
                            avg_latency += (end_time - start_time).total_seconds()
                    except Exception as e:
                        print(f"Error calculating latency for run {run.id}: {e}")

            if total_runs > 0:
                avg_latency /= total_runs

            return {
                "total_runs": total_runs,
                "error_rate": error_runs / total_runs if total_runs > 0 else 0,
                "avg_latency_seconds": avg_latency,
                "project_name": project_name
            }         
        except Exception as e:
            return{
                "error": f"Error analyzing performance: {e}",
                "total_runs": 0,
                "error_rate": 0,
                "avg_latency_seconds": 0,
                "project_name": project_name
            }       
                            
                            