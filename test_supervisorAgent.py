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
load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

db = SQLDatabase.from_uri("sqlite:///PAE.db")

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

members = ["clientAgent", "queryAgent"]

# system_prompt = (

#      "You are a supervisor able to choose between the following workers:"
# " {members}."

#     " Given the user request, decide if it is needed to generate a query or"
# "not.")

# system_prompt=f"""You are a supervisor from a car  rental company tasked with managing a conversation between the following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. If the conversation is over, respond with 'FINISH'.

# IMPORTANT OBSERVATION: The **queryAgent** manages the database interactions that contains all information from the cars from the car rental company.

#     - **clientAgent**: Use this worker for tasks that involve:
#         - Greeting and saying goodbye to the client.
#         -Asking the client for more information.
#         - Anything that has nothing to do with accessing the database.
#     - **queryAgent**: Use this worker for tasks that involve:
#         - Accessing the database.
#         - Generate SQL queries.
#         - Checking certain characteristics from cars in the database.
# Decision Criteria:
# - If the query mentions comparing cars from a list of certain cars ALREADY GIVEN , route it to the **clientAgent**.
# - If the query involves asking for certain characteristics like price, location, availability of cars, direct it to the **queryAgent**.
# - In cases where both agents could be relevant, prioritize the **queryAgent** for database fetching and **clientAgent** for immediate response to the client
# Example Decision Flow:
# - Query: "What cars do you hava available?" -> **queryAgent**
# - Query: "What are the top 3 cars available this month" -> **queryAgent**

#"""

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
# Decision Criteria:
# - If the query mentions comparing cars from a list of certain cars ALREADY GIVEN , route it to the **clientAgent**.
# - If the query involves asking for certain characteristics like price, colour,location, availability of cars, direct it to the **queryAgent**.
# - In cases where both agents could be relevant, prioritize the **queryAgent** for database fetching and **clientAgent** for immediate response to the client
# Example Decision Flow:
# - Query: "What cars do you hava available?" -> **queryAgent**
# - Query: "What are the top 3 cars available this month" -> **queryAgent**
#""""

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{messages}")

    ]
)


class Router(TypedDict):

    next: Literal["queryAgent", "clientAgent"]

chain=prompt|llm.with_structured_output(Router)
messages = [

    {"role": "system", "content": system_prompt},

    ] 

while True:
    input_=input("User: ")
    
    print(chain.invoke(input_))
    