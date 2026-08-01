import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Dossier de stockage persistant pour ChromaDB
DB_PATH = str(Path(__file__).parent.parent / "memory_db")


# Modèle d'embeddings léger
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#Client ChromaDB persistant
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("papers")

def save_paper(title: str, content: str, pdf_url: str):
    """Sauvegarde un paper dans la mémoire persistante."""
    existing = collection.get(ids=[pdf_url])
    if existing["ids"]:
        return #déjà en mémoire
    
    embedding = embedder.encode(content).tolist()
    collection.add(
        ids=[pdf_url],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"title": title, "url": pdf_url}]
    )
    print(f"💾 Paper sauvegardé en mémoire : {title[:50]}")


def search_memory(query: str, n_results: int = 2) -> str:
    """Cherche dans les papers déjà lus avec un seuil de pertinence."""
    if collection.count() == 0:
        return ""

    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        return ""

    # Filtre par seuil de distance — garde seulement les résultats pertinents
    # Distance < 0.5 = très pertinent, > 0.8 = non pertinent
    filtered = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist < 0.7:  # seuil de pertinence
            filtered.append((doc, meta, dist))

    if not filtered:
        return ""  # rien de pertinent → l'agent ira chercher sur arXiv

    output = "📚 **Papers déjà en mémoire :**\n\n"
    for doc, meta, dist in filtered:
        output += f"- **{meta['title']}**\n{doc[:300]}...\n\n"

    return output


def get_memory_count() -> int:
    """Retourne le nombre de papers en mémoire."""
    return collection.count()