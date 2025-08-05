import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Charger le modèle
model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger tes chunks
with open("pdf_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ✅ Créer une instance ChromaDB persistante
client = chromadb.PersistentClient(path="chroma_db")  # dossier où les données seront stockées

# ✅ Récupérer la collection si elle existe déjà, sinon la créer
collection = client.get_or_create_collection(name="pdf_chunks")

# Ajouter les documents (si la collection est vide)
if collection.count() == 0:
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
    print(" Embeddings ajoutés et stockés de façon persistante.")
else:
    print(" La collection contient déjà des données. Aucun ajout nécessaire.")

# Exemple de recherche
query = "Optimization"
query_embedding = model.encode(query).tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nRésultats de recherche :")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {meta['title']} (p.{meta['page']}) -> {doc[:200]}...")
