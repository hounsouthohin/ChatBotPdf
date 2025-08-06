import json
import re
import spacy
import chromadb
import ollama  # ✅ utilisation directe de l’API Ollama
import yaml  # ✅ pour charger le fichier prompts.yaml
from sentence_transformers import SentenceTransformer

# ---- Charger SpaCy
nlp = spacy.load("en_core_web_sm")

# ---- Charger le modèle Sentence Transformer
model = SentenceTransformer("./models/all-mpnet-base-v2")

# ---- Charger ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# ---- Créer (ou récupérer) une collection avec la métrique cosinus
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}  # impose la similarité cosinus (meilleure pour du texte)
)
print(" Collection prête avec cosine similarity.")

# ---- Charger tes chunks depuis le fichier JSON
with open("pdf_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ---- Réindexation seulement si la collection est vide
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

def hybrid_search(query, top_k=5, alpha=0.8):
    """
    Recherche hybride (sémantique + mots-clés).
    - alpha = poids du score sémantique (0 < alpha < 1).
    """
    # ---- Étape 1 : recherche sémantique
    query_embedding = model.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Les résultats de la première requête
    docs = semantic_results["documents"][0]
    metas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]

    # ---- Étape 2 : scoring par mots-clés + embeddings
    query_doc = nlp(query.lower())
    query_terms = {token.lemma_ for token in query_doc if token.is_alpha and not token.is_stop}
    
    ranked = []
    for i, doc in enumerate(docs):
        semantic_score = 1 - distances[i]
        doc_tokens = nlp(doc.lower())
        doc_terms = {token.lemma_ for token in doc_tokens if token.is_alpha and not token.is_stop}
        keyword_overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
        score = alpha * semantic_score + (1 - alpha) * keyword_overlap
        ranked.append((score, doc, metas[i]))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:top_k]

def generate_answer(query, context, model_name="llama3.2:latest"):
    """
    Génère une réponse avec Ollama en utilisant les passages pertinents.
    Le prompt est chargé depuis prompts.yaml pour être facilement modifiable.
    """
    # ---- Charger le prompt depuis prompts.yaml
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    base_prompt = prompts.get("pdf_assistant", "")

    # ---- Construire le prompt final
    prompt = base_prompt.format(
        question=query,
        context="\n".join(context) # les réponses pertinente.
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Erreur lors de l'appel au LLM : {e}"

# ---- Test complet
query = "The concept of Convolutional Neural Network"
results = hybrid_search(query, top_k=5, alpha=0.8)

print("\nRésultats hybrides (passages pertinents) :")
context = [] 
for score, doc, meta in results:
    subtitle = f" - {meta['subtitle']}" if meta['subtitle'] else ""
    print(f"[Score {score:.3f}] - {meta['title']}{subtitle} (p.{meta['page']}) -> {doc[:200]}...")
    context.append(doc)

print("\nRéponse finale du LLM :")
answer = generate_answer(query, context, model_name="llama3.2:latest")
print(answer)
