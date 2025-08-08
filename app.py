import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from typing import Sequence #Type générique de python

from langchain_core.messages import BaseMessage #Classe de base pour representer un message(HumainMessage,...)
from langgraph.graph.message import add_messages #Ajouter les nouveaux messages dans le state : conversation persistante.

# Annotated : permet d’attacher des métadonnées à un type parfois pour la docu.
# TypeDict : permet de définir un type dictionnaire avec des clés bien précises.
from typing_extensions import Annotated, TypedDict

import asyncio


# ---- Configurer LangSmith (optionnel, pour le traçage) avec la plateform LangSmith.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

# ---- Initialiser le modèle avec Ollama
model = init_chat_model("llama3.2:latest", model_provider="ollama")





#prompt_template : permet de rendre les prompts dynamiques i.e : Insertion de variables différentes.
prompt_template = ChatPromptTemplate.from_messages(
    #ChatPromptTemplate : Crée un schéma(user : , system : ,...) de prompt à partir de plusieurs blocs de messages
    [
        (
            "system",
            "You talk like an assistant. Answer all questions to the best of your ability in {language}, and in the way that a simple person can understand.",
        ),
        #MessagesPlacehoder : Un historique de messages sera mis sous la variable messages.
        MessagesPlaceholder(variable_name="messages"),         
    ]
)

class State(TypedDict):
    #Annotated : permet d’attacher des métadonnées à un type .
    messages: Annotated[Sequence[BaseMessage], add_messages] # ici add_messages plus qu'une métadonnées est une fonction pour gérer les messages dans le workflow.
    language: str

#Definir un nombre de token limité, avant de passer les messages au promptTemplate.
trimmer = trim_messages(
    max_tokens=65,
    strategy="last", #priorise les conversations les plus récentes.
    token_counter=model,
    include_system=True, #Inclu le systemMessage important pour maintenir le contexte.
    allow_partial=False, # Ne découpe pas les messages en morceaux
    start_on="human", # commence par le dernier message humain.
)


# Async function for node:
workflow = StateGraph(state_schema=State)
def call_model(state: State): #State au lieu de MessagesState #MessagesState : boite contenant l'historique de conversation.
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}          



# Define graph as before:

workflow.add_edge(START, "model") # Une fleche ou direction vers l'execution du noeud model
workflow.add_node("model", call_model) #quand on atteint ce noeud exécute la fonction call_model.
app = workflow.compile(checkpointer=MemorySaver())

def main():
    config = {"configurable": {"thread_id": "abc789"}}
    query = "Hi I'm Todd, please tell me a joke."
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


main()
