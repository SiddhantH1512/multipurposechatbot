from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from config import Config
from langgraph.prebuilt import ToolNode, tools_condition
from utils import ChatGrokModel, ChatGeminiModel, ChatOllamaModel
from tools.tool_list import all_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
import sqlite3
import asyncio
import requests
import aiosqlite
import threading

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# CREATING CLIENT AND CONNECTING WITH SERVER
servers = {
    "arithmetic": {
        "transport":"stdio",
        "command":"/Users/siddhant/Desktop/projects/chatbot/myvenv/bin/python",
        "args":["/Users/siddhant/Desktop/fastmcp-math-server/main.py"]
    },
    "ExpenseTracker": {
        "transport":"stdio",
        "command":"/Users/siddhant/Desktop/projects/chatbot/myvenv/bin/python",
        "args":["/Users/siddhant/Desktop/fastmcp-server/main.py"]
    }
}

client = MultiServerMCPClient(servers)

# TOOLS
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()
all_tools.extend(mcp_tools)

model = ChatGrokModel()
llm_with_tools = model.bind_tools(all_tools) if all_tools else model
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(all_tools) if all_tools else None

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)




async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())




