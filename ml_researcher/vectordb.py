"""Backend switch: VECTOR_BACKEND=chroma (default, local) or rag_engine (Vertex-managed)."""
from .config import VECTOR_BACKEND

if VECTOR_BACKEND == "rag_engine":
    from .store_rag import count,delete_paths,indexed_files,search,upsert_file
else:
    from .store_chroma import count,delete_paths,indexed_files,search,upsert_file