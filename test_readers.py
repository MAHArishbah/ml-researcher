"""Usage: python test_readers.py <file> ["search query"]"""
import sys,time
from ml_researcher.readers import read_file,chunks,to_wsl_path
from ml_researcher import store_chroma as s

p=to_wsl_path(sys.argv[1])          # accepts E:\... or /mnt/e/...
t=time.time()
text=read_file(p)
print(f"--- extracted {len(text)} chars in {time.time()-t:.1f}s")
print(f"--- image descriptions: {text.count('[Figure description')+text.count('[Image')}")
pieces=list(chunks(text))
print(f"--- {len(pieces)} chunks, size min/max: {min(map(len,pieces),default=0)}/{max(map(len,pieces),default=0)}")

for marker in ("[Figure description","[Image"):   # show first image description, else start of text
    i=text.find(marker)
    if i>=0:
        print(text[i:i+600]); break
else:
    print(text[:600])

if len(sys.argv)>2:
    print("\n--- chunks stored:",s.upsert_file(p,p.stat().st_mtime))
    for r in s.search(sys.argv[2],top_k=3):
        print(f"\n{r['name']}  sim={r['vector_similarity']}  kw={r['keyword_hit']}\n{r['text'][:200]}")
