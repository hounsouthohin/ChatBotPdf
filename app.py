
import json
import spacy
import chromadb


from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
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

#Trouver une alternative au type de model spacy : rendre dynamique
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

# --- Encodage des chunks sinon déjà fait
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

def build_where_filters(query):

    query_doc = nlp(query)
    query_terms = [token.lemma_ for token in query_doc if token.is_alpha and not token.is_stop]

    if not query_terms:
        return {}

    # Construire filtre where_document
    if len(query_terms) == 1:
        # Un seul terme, pas besoin de $or
        where_document = {"$contains": query}
    else:
        # Plusieurs termes => $or
        or_conditions_doc = [{"$contains": term} for term in query_terms]
        where_document = {"$or": or_conditions_doc}

    return where_document


#Implémentation dynamique de alpha.
def hybrid_search(query, top_k=1):
    where_document = build_where_filters(query)
    print (where_document)
    query_embedding = embedding_model.encode(query).tolist()
    #Semantic_results : les resultats de la recherche sémantique
    semantic_results = collection.query( 
        query_embeddings=[query_embedding],
        n_results=top_k,
        #where_document=where_document,
        include=["documents", "metadatas", "distances"]
    )
    # Les resultats de la première requete
    docs = semantic_results["documents"][0]
    metas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]

    #query_doc = nlp(query.lower()) # Identification des tokens de la requete
    #token.is_alpha : garde les tokens qui sont (A-Z ou a-z)
    # token.is_stop : les mots vide : the, is...
    # token.lemma_ : la forme de base du token : running - run...
    #query_terms = {token.lemma_ for token in query_doc if token.is_alpha and not token.is_stop}

    ranked = []
    # Parcours des resultats de la recherche sémantique
    for i, doc in enumerate(docs):
        if(distances[i] >= 0) :
            semantic_score = 1- distances[i] 
            ranked.append((semantic_score, doc, metas[i]))
    ranked.sort(key=lambda x: x[0], reverse=True) # Du score le plus grand au plus petit.
    return ranked[:top_k]


# --- Prompt template LangChain/LangGraph avec historique + contexte
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. You must answer the user's question using **only the information provided in the context below**.*Context:{context}.You must strictly use the language: {language} that it provided"
    ),
    MessagesPlaceholder(variable_name="messages"),
])



# --- Typage du state pour LangGraph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# ---- Initialiser le modèle avec Ollama
model = init_chat_model("llama3.2:latest", model_provider="ollama")


trimmer = trim_messages(
    max_tokens=300,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)



# --- Fonction principale du noeud LangGraph avec intégration complète
def call_model(state: State):
    # Limiter l'historique à max tokens
    trimmed_messages = trimmer.invoke(state["messages"])

    #Récupérer le dernier message humain : et si ce n'est pas une question.
    user_query = ""
    for msg in reversed(trimmed_messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
        
    #Recherche hybride dans la base
    results = hybrid_search(user_query)
    context = "\n---\n".join(doc for _, doc, _ in results)
    print("******* result ****** : ",results[:1000])
    #Construire le prompt complet avec historique, contexte et question
    final_prompt = prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "context": context,
            "language": state["language"]
        }
    )
    
    
    response = model.invoke(final_prompt)
    return {"messages": [response]}          

  

# --- Construire et compiler le workflow LangGraph
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())


def main():
    """
    Fonction principale pour interagir avec l'utilisateur et générer une réponse
    à partir du chatbot (Ollama + LangGraph).
    """

    # Message d'accueil
    print("=== Chatbot IA - Mode Streaming Propre ===")

    # 1. Récupérer la question de l'utilisateur
    question = input("Entrez votre question : ")

    # 2. Définir la langue de sortie (ici anglais par défaut)
    language = "english"

    # 3. Préparer les messages pour l'entrée du modèle
    input_messages = [
        ("user", question)  # tuple (rôle, contenu)
    ]

    # 4. Configuration du streaming
    config = {"configurable": {"thread_id": "1"}}

    # 5. Stocker la réponse complète au fur et à mesure
    full_response = ""

    # 6. Streaming des morceaux de réponse
    print("\n[Streaming en cours...]\n")
    for chunk in app.stream(
        {"messages": input_messages, "language": language},
        config,
        stream_mode="messages"
    ):
        message_chunk = chunk[0]  # AIMessageChunk
        
        content = None
        if hasattr(message_chunk, "content"):
            attr = getattr(message_chunk, "content")
            content = attr() if callable(attr) else attr
        elif hasattr(message_chunk, "text"):
            attr = getattr(message_chunk, "text")
            content = attr() if callable(attr) else attr

        if content:
            full_response += content
            print(content, end="", flush=True)




       

    # 7. Afficher la réponse complète à la fin (plus lisible)
    print("\n\n=== Réponse complète ===")
    print(full_response)


# Lancer le programme
if __name__ == "__main__":
    main()
#################################PENSER A ACCELERER L EXECUTION DU SCRIPT~################### 