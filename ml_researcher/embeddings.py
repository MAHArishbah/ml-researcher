from google import genai
from .config import PROJECT, LOCATION, EMBED_MODEL, EMBED_DIM
import numpy as np
import time
import httpx
from google.genai import errors


_client=genai.Client(vertexai=True,project=PROJECT,location=LOCATION)

def _is_retryable(e:Exception)->bool:
    if isinstance(e,errors.APIError):
        return e.code==429 or e.code >= 500
    return isinstance(e,httpx.TransportError)

def embed(text:str,kind:str) -> np.ndarray:
    """kind = 'doc' (text already prefixed with title) or 'query'. Returns a unit-norm vector."""
    payload=f"task: search result | query: {text}" if kind =="query" else text
    last_err:Exception| None =None
    for attempt in range(5):
        try:
            resp=_client.models.embed_content(
                model=EMBED_MODEL,
                contents=payload,
                config=genai.types.EmbedContentConfig(output_dimensionality=EMBED_DIM)
            )
            if not resp.embeddings or resp.embeddings[0].values is None:
                raise ValueError("empty embedding response")
            v=np.asarray(resp.embeddings[0].values,dtype=np.float32) # float 64(numpy default) -> flot32
            return v/(np.linalg.norm(v) + 1e-12)
        except Exception as e:
            if not _is_retryable(e):
                raise
            last_err=e
            if attempt <4:
                time.sleep(2 ** attempt)
    raise RuntimeError("embedding failed after 5 attempts") from last_err

