from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from config import Config
from langchain_groq import ChatGroq

model = ChatGroq(
    api_key=Config.GROQ_API_KEY,
    model_name=Config.MODEL_NAME,
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatnode(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {'messages': [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chatnode", chatnode)
graph.add_edge(START, "chatnode")
graph.add_edge("chatnode", END)

chatbot = graph.compile(checkpointer=checkpointer)