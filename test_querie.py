import json
from sentence_transformers import SentenceTransformer
import chromadb
import re
import spacy
#import subprocess, sys

# ---- Charger SpaCy (avec installation auto si nécessaire)
#try:
    #nlp = spacy.load("en_core_web_sm")
#except OSError:
    #subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    #nlp = spacy.load("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

# Charger le modèle Sentence Transformer
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Charger ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Créer (ou récupérer) une collection avec la métrique cosinus
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}  # impose la similarité cosinus (meilleure pour du texte)
)
print(" Collection prête avec cosine similarity.")

# Charger tes chunks depuis le fichier JSON
with open("pdf_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

#  Réindexation seulement si la collection est vide
if collection.count() == 0:
    print(" Collection vide → réindexation des embeddings...")
    for ch in chunks:
        # Concaténation titre + sous-titre + texte pour enrichir le contexte
        combined_text = f"{ch['title']} {ch['subtitle']} {ch['text']}".strip()
        
        embedding = model.encode(combined_text).tolist()
        collection.add(
            ids=[str(ch["chunk_id"])],
            embeddings=[embedding],
            documents=[ch["text"]],  # garde uniquement le texte brut pour l'affichage
            metadatas=[{
                "title": ch["title"],
                "subtitle": ch["subtitle"] if ch["subtitle"] else "",
                "page": ch["page"]
            }]
        )
    print("✅ Tous les embeddings ont été réindexés avec titre + sous-titre + texte.")
else:
    print("✅ Collection déjà indexée, aucun ajout nécessaire.")

def hybrid_search(query, top_k=5, alpha=0.5):
    """
    Recherche hybride (sémantique + mots-clés).
    - alpha = poids du score sémantique (0 < alpha < 1).
    """
    # ---- Étape 1 : recherche sémantique
    query_embedding = model.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,  # On récupère plus pour reranker ensuite
        include=["documents", "metadatas", "distances"]  # inutile de recalculer les embeddings
    )
    
    # Les résultats de la première requête
    docs = semantic_results["documents"][0]
    metas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]  # Plus petite distance = meilleure similarité

    # ---- Étape 2 : scoring par mots-clés + embeddings
    # Extraction des tokens essentiels via SpaCy (noms, verbes, adjectifs, etc.)
    query_doc = nlp(query.lower())
    query_terms = {token.lemma_ for token in query_doc if token.is_alpha and not token.is_stop}
    
    ranked = []
    for i, doc in enumerate(docs):
        # Score sémantique directement depuis Chroma
        semantic_score = 1 - distances[i]  # (1 = identique, 0 = très différent)
        
        # Tokens du document avec SpaCy
        doc_tokens = nlp(doc.lower())
        doc_terms = {token.lemma_ for token in doc_tokens if token.is_alpha and not token.is_stop}
        
        # Calcul du score lexical (chevauchement des mots)
        keyword_overlap = len(query_terms & doc_terms) / max(1, len(query_terms))  # Intersection des mots clés
        
        # Score final = combinaison
        # alpha proche de 1 → priorité aux embeddings (sémantique).
        # alpha proche de 0 → priorité aux mots-clés (lexical).
        score = alpha * semantic_score + (1 - alpha) * keyword_overlap
        ranked.append((score, doc, metas[i]))

    # ---- Étape 3 : trier et retourner top_k
    ranked.sort(key=lambda x: x[0], reverse=True)  # les scores sont triés du plus grand au plus petit
    return ranked[:top_k]

#  Test
query = "From Model to production?"
results = hybrid_search(query, top_k=5, alpha=0.7)

print("\nRésultats hybrides :")
for score, doc, meta in results:
    subtitle = f" - {meta['subtitle']}" if meta['subtitle'] else ""
    print(f"[Score {score:.3f}] - {meta['title']}{subtitle} (p.{meta['page']}) -> {doc[:200]}...")
