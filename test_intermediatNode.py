from langchain_groq import ChatGroq 
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm =  ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

promptQuery = """You are an employee in a car rental company and you are in charge to interpret the input of the user IN NATURAL LANGUAGE, NOT SQL COMMANDS, and then 
return a request with the characteristics of the cars that need to be found from the database. Only return a single request as the output do NOT give explanation to your output. 
Do NOT give more than one output. 

Use the following format:

Output: give me cars...
"""

promptQueryV2= """You are an employee in a car rental company and your job is to interpret the user's  question and refrase it so that it is an order to search. 
The order should always start with "give me a car...". Only return a single order as the output and do NOT give explanations to your request. 

The information you have about the car rental company is the following:

 - Cars are priced from 20.4 to 199.72 price/day being 20.4 the lowest to 199.72 the highest.

Use the following format:

Output: give me cars... 
"""

prompt_template = ChatPromptTemplate([
    ("system", promptQueryV2),
    ("user", "{input_}")
])

while True:
    userinput=input("user:")
    agent = prompt_template | llm
    print(agent.invoke({"input_": userinput}).content)