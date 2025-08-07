import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import asyncio


# ---- Configurer LangSmith (optionnel, pour le traçage) avec la plateform LangSmith.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

# ---- Initialiser le modèle avec Ollama
model = init_chat_model("llama3.2:latest", model_provider="ollama")


# Async function for node:
async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"]) # Execution du model avec la methode asynchrone
    return {"messages": response}


# Define graph as before:
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "abc123"}} #pour identifier le thread de la conversation(une file de conversation parmis tant d'autres)




async def main():
    query = "Hi! my name is Arthur"
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    
    query2 = "What is my name?"
    input_messages = [HumanMessage(query2)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

# Appel de la fonction async principale
asyncio.run(main())
