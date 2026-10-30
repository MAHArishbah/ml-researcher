import base64,hashlib,io,json
from pathlib import Path
from PIL import Image
from langchain_core.messages import HumanMessage
from .config import VISION_MODEL
from .llm import text_of
import threading

_cache_lock=threading.Lock()
CACHE=Path.home()/".ml_researcher"/"captions.json"
_cache:dict[str,str]=json.loads(CACHE.read_text()) if CACHE.exists() else {}
_OK_MIME={"image/png","image/jpeg","image/webp"}
MIN_PX=120  # skip icons/bullets smaller than this on either side

DESCRIBE=("You are indexing study notes for search. Describe this image so the description can stand in for it in a text index. "
          "For diagrams/charts/figures: say what it shows, labels/axes, key values and the main takeaway. "
          "Write equations as LaTeX and copy any visible text verbatim. "
          "If it is purely decorative (logo, icon, background), reply exactly: SKIP\n\n"
          "Surrounding text for context:\n{ctx}")
OCR=("This is a scanned page. Transcribe all text exactly, keeping paragraphs and lists. "
     "Write equations as LaTeX. Briefly describe any figures in [brackets] where they appear.")

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
    prompt=OCR if ocr else DESCRIBE.format(ctx=ctx[:1500])
    msg=HumanMessage(content=[{"type":"image","base64":base64.b64encode(img).decode(),"mime_type":mime},
                              {"type":"text","text":prompt}])
    text=text_of(_vision_llm().invoke([msg])).strip()
    text="" if text=="SKIP" else text
    with _cache_lock:
        _cache[key]=text
        CACHE.parent.mkdir(parents=True,exist_ok=True)
        CACHE.write_text(json.dumps(_cache))
    return text
