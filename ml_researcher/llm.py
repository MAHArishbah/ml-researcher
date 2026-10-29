from langchain_google_genai import ChatGoogleGenerativeAI
from .config import LOCATION,MODEL,PROJECT,RESEARCHER_MODEL

def gemini(model:str,thinkin_level:str |None=None)->ChatGoogleGenerativeAI:
    kwargs = {"model": model, "vertexai": True, "project": PROJECT,
              "location": LOCATION, "max_retries": 3}
    if thinkin_level:
        kwargs["thinking_level"]=thinkin_level
    return ChatGoogleGenerativeAI(**kwargs)

MAIN_LLM=gemini(MODEL)
FAST_LLM=gemini(MODEL,thinkin_level='low')
RESEARCH_LLM=gemini(RESEARCHER_MODEL,thinkin_level="low") # tool-calling researchers

def text_of(msg)->str:
    """Plain text of a LangChain message (Gemini content may be a list of blocks).""" 
    t=getattr(msg,"text",None)
    t=t() if callable(t) else t
    return t if isinstance(t,str) and t else str(msg.content)   
