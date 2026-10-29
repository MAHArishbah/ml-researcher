"""BM 25 keyword index + RRF"""
import re
from rank_bm25 import BM25Okapi

_TOKEN=re.compile(r"[a-z0-9]+") # "Fokker-Planck" -> ["fokker", "planck"]

def tokenize(text:str)->list:
    return _TOKEN.findall(text.lower())

class KeywordIndex:
    def __init__(self,ids,docs):
        self.ids=ids
        self.bm25=BM25Okapi([tokenize(d) for d in docs] )if docs else None

    def top(self,query:str,n:int)->list:
        """IDs of the n best keyword matches (only chunks sharing at least one term)."""
        if self.bm25 is None:
            return []
        scores=self.bm25.get_scores(tokenize(query))
        order=scores.argsort()[::-1][:n]
        return [self.ids[i] for i in order if scores[i]>0]

def rrf(rankings,k:int=60)->list:
    """Fuse ranked ID lists by rank: score(d) = sum over lists of 1/(k + rank)."""
    fused={}
    for ranking in rankings:
        for rank,doc_id in enumerate(ranking,start=1):
            fused[doc_id]=fused.get(doc_id,0.0) + 1/(k+rank)
    return sorted(fused,key=fused.get,reverse=True)
