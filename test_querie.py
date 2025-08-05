import json
from sentence_transformers import SentenceTransformer
import chromadb
import re

# Charger le modèle
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Charger ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Supprimer l’ancienne collection si elle existe
#try:
    #client.delete_collection("pdf_chunks")
    #print(" Ancienne collection supprimée.")
#except Exception:
    #print(" Aucune ancienne collection à supprimer.")

# Créer une nouvelle collection avec cosine similarity
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}  # impose la similarité cosinus
)
print(" Nouvelle collection créée avec cosine similarity.")

# Charger tes chunks
with open("pdf_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Ajouter les documents (réindexer tous les chunks)
for ch in chunks:
    embedding = model.encode(ch["text"]).tolist()
    collection.add(
        ids=[str(ch["chunk_id"])],
        embeddings=[embedding],
        documents=[ch["text"]],
        metadatas=[{
            "title": ch["title"],
            "subtitle": ch["subtitle"] if ch["subtitle"] else "",
            "page": ch["page"]
        }]
    )
print("📌 Tous les embeddings ont été réindexés.")

def hybrid_search(query, top_k=5, alpha=0.7):
    """
    Recherche hybride (sémantique + mots-clés).
    - alpha = poids du score sémantique (0 < alpha < 1).
    """
    # ---- Étape 1 : recherche sémantique
    query_embedding = model.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3,  # On récupère plus pour reranker ensuite
        include=["documents", "metadatas", "distances"]  # ✅ inutile de recalculer les embeddings
    )
    
    docs = semantic_results["documents"][0]
    metas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]  # Plus petite distance = meilleure similarité

    # ---- Étape 2 : scoring par mots-clés + embeddings
    query_terms = set(re.findall(r"\w+", query.lower())) 
    ranked = []
    for i, doc in enumerate(docs):
        #  Score sémantique directement depuis Chroma
        semantic_score = 1 - distances[i]  # (1 = identique, 0 = très différent)
        
        # Calcul du score lexical (chevauchement des mots)
        doc_terms = set(re.findall(r"\w+", doc.lower()))
        keyword_overlap = len(query_terms & doc_terms) / max(1, len(query_terms)) #Intersection des mots clés.
        
        # Score final = combinaison
        # alpha proche de 1 → priorité aux embeddings (sémantique).
        # alpha proche de 0 → priorité aux mots-clés (lexical).
        score = alpha * semantic_score + (1 - alpha) * keyword_overlap
        ranked.append((score, doc, metas[i]))

    # ---- Étape 3 : trier et retourner top_k
    ranked.sort(key=lambda x: x[0], reverse=True)  # les scores sont triés du plus grand au plus petit
    return ranked[:top_k]

# 🔎 Test
query = "optimization techniques in deep learning"
results = hybrid_search(query, top_k=5, alpha=0.7)

print("\nRésultats hybrides :")
for score, doc, meta in results:
    print(f"[Score {score:.3f}] - {meta['title']} (p.{meta['page']}) -> {doc[:200]}...")
