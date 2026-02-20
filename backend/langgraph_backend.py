from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from config import Config
from langgraph.prebuilt import ToolNode, tools_condition
from utils import ChatGrokModel
from tools.tool_list import all_tools
import sqlite3

model = ChatGrokModel()
llm_with_tools = model.bind_tools(all_tools)
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatnode(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(all_tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)


graph.add_node("chatnode", chatnode)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatnode")
graph.add_conditional_edges("chatnode", tools_condition)
graph.add_edge("tools", "chatnode")


chatbot = graph.compile(checkpointer=checkpointer)

