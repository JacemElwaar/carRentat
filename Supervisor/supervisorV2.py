import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing import Literal
from SQLagentv3 import agent_SQL
from intermediateNode import intermediate_node
load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

db = SQLDatabase.from_uri("sqlite:///PAE.db")

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

members = ["clientAgent", "queryAgent"]

system_prompt = system_prompt=f"""You are a supervisor from a car  rental company tasked with managing a conversation between the following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. If the conversation is over, respond with 'FINISH'.

IMPORTANT OBSERVATION: The **queryAgent** manages the database interactions that contains all information from the cars from the car rental company.
    - **queryAgent**: Use this worker for tasks that involve:
        - Accessing the car rental company's database.
        -Search for car characteristics like, price, location, availability, colour....
    - **clientAgent**: Use this worker for tasks that involve:
        - Greeting the client.
        -Asking the client for more information.
        -No need for database interaction
    Example Decision Flow:
    -Query:"Hi" -> **clientAgent**

    """
# Decision Criteria:
# - If the query mentions comparing cars from a list of certain cars ALREADY GIVEN , route it to the **clientAgent**.
# - If the query involves asking for certain characteristics like price, colour,location, availability of cars, direct it to the **queryAgent**.
# - In cases where both agents could be relevant, prioritize the **queryAgent** for database fetching and **clientAgent** for immediate response to the client
# Example Decision Flow:
# - Query: "What cars do you hava available?" -> **queryAgent**
# - Query: "What are the top 3 cars available this month" -> **queryAgent**
#""""

promptA=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{messages}")

    ]
)

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]

    next: str

class Router(TypedDict):

    next: Literal["queryAgent", "clientAgent"]

workflow = StateGraph(State)

# def supervisor_agent(state: State):
#     messages = [
#     {"role": "system", "content": system_prompt},
#     ] + state["messages"]
#     response = llm.with_structured_output(Router).invoke(messages)
#     next_ = response["next"]
#     return {"next": next_}

def supervisor_agentv2(state: State):
    messages = state["messages"][-1].content
    superA=promptA|llm.with_structured_output(Router)
    try:
        response=superA.invoke(messages)
        next_=response["next"]
    except: next_="clientAgent"
    return{"next": next_}

client_description = """You are an assertive customer service agent. Respond to the customer's inquiries
with the information available to you based on the input question if you can."""

client_prompt = ChatPromptTemplate.from_messages(

    [("system", client_description), ("placeholder", "{messages}")])

def client_agent(state: State):

    prompted_llm = client_prompt | llm

    result = prompted_llm.invoke(state)

    return {"messages": [result]}


workflow.add_node("supervisor", supervisor_agentv2)
workflow.add_node("queryAgent2", agent_SQL)
workflow.add_node("clientAgent", client_agent)
workflow.add_node("queryAgent", intermediate_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state["next"])
workflow.add_edge("queryAgent", "queryAgent2")
workflow.add_edge("queryAgent2", "clientAgent")
workflow.add_edge("clientAgent", END)

graph = workflow.compile() 



while True:
    user_input = input("user: ")
    for event in graph.stream({"messages": [("user", user_input)]}):
        #print(event)
        if 'clientAgent' in event:
            client_agent_response = event['clientAgent']
            print("Client's agent response:", client_agent_response['messages'][0].content)