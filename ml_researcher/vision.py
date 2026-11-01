import base64,hashlib,io,json
from pathlib import Path
from PIL import Image
from langchain_core.messages import HumanMessage
from .config import VISION_MODEL
from .llm import text_of
from .prompts import prompt
import threading

_cache_lock=threading.Lock()
CACHE=Path.home()/".ml_researcher"/"captions.json"
_cache:dict[str,str]=json.loads(CACHE.read_text()) if CACHE.exists() else {}
_OK_MIME={"image/png","image/jpeg","image/webp"}
MIN_PX=120  # skip icons/bullets smaller than this on either side

OCR=prompt("VISION_OCR")

def prepare(blob:bytes,mime:str)->tuple[bytes,str]|None:
    """Return (bytes, mime) Gemini accepts, or None for tiny/unsupported images (e.g. EMF/WMF)."""
    try:
        im=Image.open(io.BytesIO(blob))
    except Exception:
        return None
    if min(im.size)<MIN_PX:
        return None
    if mime in _OK_MIME:
        return blob,mime
    buf=io.BytesIO()
    try:
        im.convert("RGB").save(buf,"PNG")
    except Exception:            # e.g. WMF/EMF: Pillow opens the header but can't decode it on Linux
        return None
    return buf.getvalue(),"image/png"

_llm=None

def _vision_llm():
    """Any multimodal chat model (VISION_MODEL); created on first use so imports stay cheap.
    Retries and the shared rate limit come from llm.chat_model."""
    global _llm
    if _llm is None:
        from .llm import chat_model
        _llm=chat_model(VISION_MODEL)
    return _llm

def caption(img:bytes,mime:str,ctx:str="",ocr:bool=False)->str:
    key=hashlib.sha256(img).hexdigest()+(":ocr" if ocr else "")   # provider-independent: cache survives a switch
    if key in _cache:
        return _cache[key]
    instruction=OCR if ocr else prompt("VISION_DESCRIBE",ctx=ctx[:1500])
    msg=HumanMessage(content=[{"type":"image","base64":base64.b64encode(img).decode(),"mime_type":mime},
                              {"type":"text","text":instruction}])
    text=text_of(_vision_llm().invoke([msg])).strip()
    text="" if text=="SKIP" else text
    with _cache_lock:
        _cache[key]=text
        CACHE.parent.mkdir(parents=True,exist_ok=True)
        CACHE.write_text(json.dumps(_cache))
    return text
