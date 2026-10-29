import hashlib,io,json,time
from pathlib import Path
from PIL import Image
from google import genai
from .config import RESEARCHER_MODEL as MODEL
from .embeddings import _client,_is_retryable
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
    im.convert("RGB").save(buf,"PNG")
    return buf.getvalue(),"image/png"

def caption(img:bytes,mime:str,ctx:str="",ocr:bool=False)->str:
    key=hashlib.sha256(img).hexdigest()+(":ocr" if ocr else "")
    if key in _cache:
        return _cache[key]
    prompt=OCR if ocr else DESCRIBE.format(ctx=ctx[:1500])
    last_err:Exception|None=None
    for attempt in range(5):
        try:
            resp=_client.models.generate_content(
                model=MODEL,
                contents=[genai.types.Part.from_bytes(data=img,mime_type=mime),prompt],
            )
            text=(resp.text or "").strip()
            text="" if text=="SKIP" else text
            with _cache_lock:
                _cache[key]=text
                CACHE.parent.mkdir(parents=True,exist_ok=True)
                CACHE.write_text(json.dumps(_cache))
            return text
        except Exception as e:
            if not _is_retryable(e):
                raise
            last_err=e
            if attempt<4:
                time.sleep(2**attempt)
    raise RuntimeError("caption failed after 5 attempts") from last_err
