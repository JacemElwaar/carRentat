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
from intermediateNode import intermidiate_query
load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

db = SQLDatabase.from_uri("sqlite:///PAE.db")

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

members = ["clientAgent", "queryAgent"]

system_prompt=f"""You are a supervisor from a car  rental company tasked with managing a conversation between the following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. If the conversation is over, respond with 'FINISH'.

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
super_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{messages}")
    ]
)
class State(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]

    next: str

class Router(TypedDict):

    next: Literal["queryAgent", "clientAgent"]

workflow = StateGraph(State)

def supervisor_agent(state: State):

    messages=state["messages"][0].content

    chain=super_prompt|llm.with_structured_output(Router)
    try:

        response = chain.invoke(messages)
    
        next_ = response["next"]
        print(next_)
    except:
        next_ = "clientAgent"

    return {"next": next_}



client_description = """You are an assertive customer service agent. Respond to the customer's inquiries
with the information available to you based on the input question if you can."""

client_prompt = ChatPromptTemplate.from_messages(

    [("system", client_description), ("placeholder", "{messages}")])

def client_agent(state: State):

    prompted_llm = client_prompt | llm

    result = prompted_llm.invoke(state)

    return {"messages": [result]}


workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("queryAgent", intermidiate_query)
workflow.add_node("SQLAgent", agent_SQL)
workflow.add_node("clientAgent", client_agent)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state["next"])
#workflow.add_conditional_edges("supervisor", ["clientAgent","queryAgent"])
workflow.add_edge("queryAgent", "SQLAgent")
workflow.add_edge("SQLAgent", "clientAgent")
workflow.add_edge("clientAgent", END)


# from langgraph.checkpoint.memory import MemorySaver

# memory = MemorySaver()

# graph = workflow.compile(checkpointer=memory) 

# config = {"configurable": {"thread_id": "1"}}


graph = workflow.compile() 

from langchain_core.runnables.graph import MermaidDrawMethod

# Generate the PNG data
png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

# Save to a file
file_path = 'Super_act.png'
with open(file_path, 'wb') as f:
    f.write(png_data)

# Automatically open the saved image (Windows)
os.startfile(file_path)

while True:
    user_input = input("user: ")
    for event in graph.stream({"messages": [("user", user_input)]}):
        #print(event)
        if 'clientAgent' in event:
            client_agent_response = event['clientAgent']
            print("Client's agent response:", client_agent_response['messages'][0].content)