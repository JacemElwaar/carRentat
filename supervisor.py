import os
from dotenv import load_dotenv
from langchain_community.tools.sql_database.tool import (

InfoSQLDatabaseTool,

ListSQLDatabaseTool,

QuerySQLDataBaseTool,

)
from SQLagentv3 import agentSQL

from langchain_community.utilities import SQLDatabase

from langchain_core.prompts import ChatPromptTemplate

from typing import Annotated

from langchain_groq import ChatGroq

from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

from langgraph.graph.message import AnyMessage, add_messages

from langgraph.prebuilt import ToolNode

from typing import Literal

load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

db = SQLDatabase.from_uri("sqlite:///PAE.db")

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

list_tool = ListSQLDatabaseTool(db=db)

info_tool = InfoSQLDatabaseTool(db=db)

query_tool = QuerySQLDataBaseTool(db=db)

from langchain_core.runnables.graph import MermaidDrawMethod


class State(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]

    next: str

workflow = StateGraph(State)

def list_tables_tool_call(state: State):

    return {"messages": [list_tool.invoke("")]}

workflow.add_node("list_tables_tool", list_tables_tool_call)

def info_schema(state: State):

    return {"messages": [info_tool.invoke(state["messages"][-1].content)]}

workflow.add_node("model_get_schema", info_schema)

query_tool_node = ToolNode(tools=[query_tool])

workflow.add_node("query_tool", query_tool_node)

members = ["clientAgent", "queryAgent"]

system_prompt = (

     "You are a supervisor able to choose between the following workers:"
" {members}."

    " Given the user request, decide if it is needed to generate a query or"
"not.")

class Router(TypedDict):

    next: Literal["queryAgent", "clientAgent"]

def supervisor_agent(state: State):
    messages = [
    {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    return {"next": next_}

workflow.add_node("supervisor", supervisor_agent)

description = """You are an SQLite expert. Generate a valid SQL query based on the input question. YOU
MUST USE QuerySQLDataBaseTool and return the result, the limit of rows you can extract is 10."""

prompt = ChatPromptTemplate.from_messages(

    [("system", description), ("placeholder", "{messages}")])

def query_agent(state: State):

    llm_with_tool = llm.bind_tools([query_tool])

    prompted_llm = prompt | llm_with_tool

    return {"messages": [prompted_llm.invoke(state)]}

workflow.add_node("queryAgent", query_agent)

client_description = """You are an assertive customer service agent. Respond to the customer's inquiries
with the information available to you based on the input question if you can."""

client_prompt = ChatPromptTemplate.from_messages(

    [("system", client_description), ("placeholder", "{messages}")])

def client_agent(state: State):

    prompted_llm = client_prompt | llm

    result = prompted_llm.invoke(state)

    return {"messages": [result]}

workflow.add_node("clientAgent", client_agent)

workflow.add_edge(START, "list_tables_tool")

workflow.add_edge("list_tables_tool", "model_get_schema")

workflow.add_edge("model_get_schema", "supervisor")

workflow.add_conditional_edges("supervisor", lambda state: state["next"])

workflow.add_edge("queryAgent", "query_tool")

workflow.add_edge("query_tool", "clientAgent")

workflow.add_edge("clientAgent", END)

graph = workflow.compile()


while True:

    user_input = input("User: ")

    for event in graph.stream({"messages": [("user", user_input)]}):

        #print(event)
        if 'client_agent' in event:
            client_agetn_response = event['client_agent']
            print("Client's agent response:", client_agetn_response['messages'][0].content)










