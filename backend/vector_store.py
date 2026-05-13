"""
Phase 2: Vector Store
Uses ChromaDB for local semantic search — no cloud needed.
Falls back to keyword search if ChromaDB not available.
"""
import os
from datetime import datetime
from typing import List, Dict

USE_CHROMA = True
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    USE_CHROMA = False
    print("ChromaDB not installed — using keyword fallback. Run: pip install chromadb")


class VectorStore:
    def __init__(self):
        self.client = None
        self.ef = None
        self._collections = {}

    def init(self):
        if not USE_CHROMA:
            print("Vector store: keyword fallback mode")
            return

        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            # Use OpenAI embeddings if available, else default
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                self.ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_key,
                    model_name="text-embedding-3-small"
                )
            else:
                self.ef = embedding_functions.DefaultEmbeddingFunction()

            print("Vector store: ChromaDB ready")
        except Exception as e:
            print(f"ChromaDB init failed: {e}")
            self.client = None

    def _get_collection(self, user_id: str):
        if not self.client:
            return None
        safe_id = user_id.replace("-", "_").replace(" ", "_")[:50]
        col_name = f"memories_{safe_id}"
        if col_name not in self._collections:
            try:
                if self.ef:
                    self._collections[col_name] = self.client.get_or_create_collection(
                        name=col_name,
                        embedding_function=self.ef
                    )
                else:
                    self._collections[col_name] = self.client.get_or_create_collection(name=col_name)
            except Exception as e:
                print(f"Collection error: {e}")
                return None
        return self._collections[col_name]

    def add_entry(self, user_id: str, entry_id: int, text: str, timestamp: str):
        col = self._get_collection(user_id)
        if not col:
            return
        try:
            col.upsert(
                ids=[str(entry_id)],
                documents=[text[:1000]],
                metadatas=[{"entry_id": entry_id, "date": timestamp[:10], "timestamp": timestamp}]
            )
        except Exception as e:
            print(f"Vector add failed: {e}")

    def search(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        col = self._get_collection(user_id)
        if not col:
            return []
        try:
            count = col.count()
            if count == 0:
                return []
            results = col.query(
                query_texts=[query[:500]],
                n_results=min(top_k, count)
            )
            out = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if "distances" in results else 0
                    out.append({
                        "entry_id": meta.get("entry_id", int(doc_id)),
                        "text": results["documents"][0][i][:200],
                        "date": meta.get("date", ""),
                        "score": round(1 - distance, 3)
                    })
            return out
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

    def clear_user(self, user_id: str):
        safe_id = user_id.replace("-", "_").replace(" ", "_")[:50]
        col_name = f"memories_{safe_id}"
        if not self.client:
            return
        try:
            self.client.delete_collection(col_name)
            self._collections.pop(col_name, None)
        except:
            pass


vs = VectorStore()
