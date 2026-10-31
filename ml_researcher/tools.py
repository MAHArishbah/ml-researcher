import arxiv
import requests
from googleapiclient.discovery import build
import html

from . import codeindex, vectordb
from .config import GITHUB_TOKEN,YOUTUBE_API_KEY

def arxiv_search(query:str,max_results: int =5) ->dict:
    """Search arXiv for research papers.

    Args:
        query: arXiv query. Supports field prefixes like ti: and abs: with AND/OR,
            e.g. 'abs:"flow matching" AND abs:"score matching"', or plain keywords.
        max_results: number of papers to return (1-10).
    """
    try:
        search=arxiv.Search(query=query,max_results=min(max_results,10),sort_by=arxiv.SortCriterion.Relevance)
        papers = [{"title": r.title,
                   "authors": ", ".join(a.name for a in r.authors[:4]),
                   "year": r.published.year,
                   "url": r.entry_id,
                   "abstract": r.summary[:800]}
                  for r in arxiv.Client().results(search)]
        return {"status":"ok","papers":papers}
    except Exception as e:
        return {"status": "error", "message": str(e)}   

def search_local_notes(query: str, top_k: int = 5, file_type: str = "") -> dict:
    """Search the user's own PDFs, Word documents, Markdown notes, power point presentations (semantic + keyword).

    Args:
        query: description of the concept; keep exact jargon and acronyms.
        top_k: number of passages to return (1-10).
        file_type: optional filter, one of 'pdf', 'docx', 'doc', 'md','pptx','ppt'. Empty = everything.
    """
    try:
        if vectordb.count() == 0:
            return {"status": "error",
                    "message": "Notes index is empty. Run: python -m ml_researcher.ingest <folder>"}
        return {"status": "ok", "hits": vectordb.search(query, min(top_k, 10), file_type)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_my_code(query: str, top_k: int = 5, language: str = "") -> dict:
    """Search the user's own code (their projects: Python files, notebooks, scripts, configs).
    Each hit is a whole function, class or group of notebook cells, with its file path, symbol and line.

    Args:
        query: what the code does or implements; use names of functions, libraries and ML terms.
        top_k: number of code chunks to return (1-10).
        language: optional file extension filter, e.g. 'py', 'ipynb', 'ts', 'md'. Empty = all code.
    """
    try:
        if codeindex.count() == 0:
            return {"status": "error",
                    "message": "Code index is empty. Run: python -m ml_researcher.ingest --code"}
        return {"status": "ok", "hits": codeindex.search(query, min(top_k, 10), language)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def github_search(query: str, max_results: int = 5) -> dict:
    """Search GitHub repositories for implementations of an ML idea.

    Args:
        query: GitHub search syntax, e.g. 'flow matching language:python stars:>20'.
        max_results: number of repositories to return (1-10).
    """
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get("https://api.github.com/search/repositories",
                         params={"q": query, "per_page": min(max_results, 10)},
                         headers=headers, timeout=20)
        r.raise_for_status()
        repos = [{"name": it["full_name"], "url": it["html_url"],
                  "stars": it["stargazers_count"],
                  "description": (it["description"] or "")[:300],
                  "language": it["language"], "updated": it["pushed_at"][:10]}
                 for it in r.json().get("items", [])]
        return {"status": "ok", "repos": repos}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def youtube_search(query: str, max_results: int = 5) -> dict:
    """Search YouTube for explanatory videos: lectures, tutorials, paper walkthroughs.

    Args:
        query: search keywords, e.g. 'flow matching explained lecture'.
        max_results: number of videos to return (1-10).
    """
    try:
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        res = yt.search().list(q=query, part="snippet", type="video",
                               maxResults=min(max_results, 10),
                               relevanceLanguage="en").execute()
        videos = [{"title": html.unescape(it["snippet"]["title"]),
                   "channel": it["snippet"]["channelTitle"],
                   "url": f"https://www.youtube.com/watch?v={it['id']['videoId']}",
                   "published": it["snippet"]["publishedAt"][:10],
                   "description": html.unescape(it["snippet"]["description"])[:300]}
                  for it in res.get("items", [])]
        return {"status": "ok", "videos": videos}
    except Exception as e:
        return {"status": "error", "message": str(e)}
