
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from intermediateNode import State


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

summarizer_description = """You are a very good sumarizer that works on a car rental company, your job is to make a concise summary 
of the events that have ocurred during each convesation between the human and the ChatBot. 
USE THE FOLLOWING CHAT
Human: """

summarizer_prompt = ChatPromptTemplate.from_messages(

    [("system", summarizer_description), ("placeholder", "{messages}")])

def memory(state: State):
    messages=state["messages"]
    prompted_llm = summarizer_prompt | llm
    result = prompted_llm.invoke(state)
    return {"messages": [result]}