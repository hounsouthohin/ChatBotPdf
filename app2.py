import os
import json
import spacy
import chromadb
import ollama

from sentence_transformers import SentenceTransformer

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import Sequence
from typing_extensions import Annotated, TypedDict

# --- Chargement SpaCy + modèle embeddings
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("./models/all-mpnet-base-v2")

# --- Initialiser ChromaDB et collection
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"},
)

# --- Charger les chunks depuis JSON
with open("pdf_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Réindexation si nécessaire
if collection.count() == 0:
    print("Réindexation des embeddings...")
    for ch in chunks:
        combined_text = f"{ch['title']} {ch.get('subtitle', '')} {ch['text']}".strip()
        embedding = embedding_model.encode(combined_text).tolist()
        collection.add(
            ids=[str(ch["chunk_id"])],
            embeddings=[embedding],
            documents=[ch["text"]],
            metadatas=[{
                "title": ch["title"],
                "subtitle": ch.get("subtitle", ""),
                "page": ch["page"]
            }]
        )
    print("Réindexation terminée.")
else:
    print("Collection déjà indexée.")


# --- Fonction recherche hybride (cosine + mots clés)
def hybrid_search(query, top_k=5, alpha=0.8):
    query_embedding = embedding_model.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = semantic_results["documents"][0]
    metas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]

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


# --- Prompt template LangChain/LangGraph avec historique + contexte
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who answers questions clearly in {language}. Use simple language.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Context:\n{context}\nUse this information to answer the user question accurately."
        ),
        (
            "user",
            "{question}"
        )
    ]
)


# --- Typage du state pour LangGraph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


def simple_token_counter(messages):
    # compte le nombre total de mots dans tous les messages (approximation)
    return sum(len(m.content.split()) for m in messages)

trimmer = trim_messages(
    max_tokens=300,
    strategy="last",
    token_counter=simple_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)



# --- Fonction principale du noeud LangGraph avec intégration complète
def call_model(state: State):
    # 1) Limiter l'historique à max tokens
    trimmed_messages = trimmer.invoke(state["messages"])

    # 2) Extraire la dernière question humaine dans l'historique réduit
    query = next(
        (m.content for m in reversed(trimmed_messages) if isinstance(m, HumanMessage)),
        ""
    )
    language = state["language"]

    # 3) Recherche hybride dans la base
    results = hybrid_search(query, top_k=5, alpha=0.8)
    context = "\n---\n".join(doc for _, doc, _ in results)

    # 4) Construire le prompt complet avec historique, contexte et question
    prompt = prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "context": context,
            "question": query,
            "language": language
        }
    )

    # 5) Préparer les messages pour Ollama (liste dict role/content)
    # On convertit le prompt LangChain en messages Ollama
    # Ici on découpe en "system" et "user" selon la structure de prompt_template

    # prompt_template crée une liste de messages sous format ChatMessage (role, content).
    # LangChain ne renvoie pas directement ce format, donc on va parser :
    # prompt_template.invoke retourne un string complet, on préfère faire la conversion manuelle :

    messages_ollama = []

    # Ajout du premier message système
    messages_ollama.append({"role": "system", "content": f"You are a helpful assistant speaking {language}."})

    # Le message utilisateur contient le prompt complet (historique + contexte + question)
    # ici on passe tout dans un seul message utilisateur pour simplifier
    messages_ollama.append({"role": "user", "content": prompt})

    # 6) Appel à Ollama
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=messages_ollama,
        )
        answer = response["message"]["content"].strip()
    except Exception as e:
        answer = f"Erreur lors de l'appel au LLM : {e}"

    # 7) Retourner la réponse sous forme AIMessage pour LangGraph
    return {"messages": [AIMessage(content=answer)]}


# --- Construire et compiler le workflow LangGraph
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())


# --- Fonction principale pour tester
def main():
    config = {"configurable": {"thread_id": "abc789"}}
    query = "What is a Convolutional Neural Network?"
    language = "English"

    input_messages = [HumanMessage(query)]
    response = ""
    for chunk in app.stream(
        {"messages": input_messages, "language": language},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            response += chunk.content

    print("Réponse complète :", response)


if __name__ == "__main__":
    main()
