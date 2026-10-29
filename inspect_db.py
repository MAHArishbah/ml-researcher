"""Usage: python inspect_db.py ["search query"]"""
import sys
from collections import Counter
from ml_researcher import store_chroma as s

col=s._collection()
print("chunks:",col.count())

metas=col.get(include=["metadatas"])["metadatas"]
files=Counter(m["name"] for m in metas)
print("files:",len(files),"| by type:",dict(Counter(m["ext"] for m in metas)))
for name,n in files.most_common(10):
    print(f"  {n:4d} chunks  {name}")

peek=col.get(where={"name":"fisher-information-consolidated-notes.pdf"},limit=2,include=["documents","metadatas"])
for d,m in zip(peek["documents"],peek["metadatas"],strict=True):
    print(f"\n--- {m['name']} chunk {m['chunk']} ---\n{d[:400]}")

if len(sys.argv)>1:
    for r in s.search(sys.argv[1],top_k=5):
        print(f"\n{r['name']}  sim={r['vector_similarity']}  kw={r['keyword_hit']}\n{r['text'][:200]}")
