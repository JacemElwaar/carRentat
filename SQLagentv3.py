import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional, TypedDict, Union
from langgraph.graph.message import AnyMessage, add_messages

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



db = SQLDatabase.from_uri("sqlite:///PAE.db")
#print(db.dialect)
table_info = db.get_usable_table_names()


llm =  ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]

    next: str




def _strip(text: str) -> str:
    return text.replace('"','').replace('SQLQuery: ','').replace('`', '').replace('´', '').replace('sql', '').strip()
    #return "SELECT COUNT(*) FROM Employee"

   
class SQLInput(TypedDict):
    """Input for a SQL Chain."""

    question: str

class SQLInputWithTables(TypedDict):
    """Input for a SQL Chain."""

    question: str
    table_names_to_use: List[str]



def create_sql_query_chain_v2(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]:
    """Create a chain that generates SQL queries.

    *Security Note*: This chain generates SQL queries for the given database.

        The SQLDatabase class provides a get_table_info method that can be used
        to get column information as well as sample data from the table.

        To mitigate risk of leaking sensitive data, limit permissions
        to read and scope to the tables that are needed.

        Optionally, use the SQLInputWithTables input type to specify which tables
        are allowed to be accessed.

        Control access to who can submit requests to this chain.

        See https://python.langchain.com/docs/security for more information.

    Args:
        llm: The language model to use.
        db: The SQLDatabase to generate the query for.
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect. Defaults to None. See Prompt section below for more.
        k: The number of results per select statement to return. Defaults to 5.

    Returns:
        A chain that takes in a question and generates a SQL query that answers
        that question.

    Example:

        .. code-block:: python

            # pip install -U langchain langchain-community langchain-openai
            from langchain_openai import ChatOpenAI
            from langchain.chains import create_sql_query_chain
            from langchain_community.utilities import SQLDatabase

            db = SQLDatabase.from_uri("sqlite:///Chinook.db")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            chain = create_sql_query_chain(llm, db)
            response = chain.invoke({"question": "How many employees are there"})

    Prompt:
        If no prompt is provided, a default prompt is selected based on the SQLDatabase dialect. If one is provided, it must support input variables:
            * input: The user question plus suffix "\nSQLQuery: " is passed here.
            * top_k: The number of results per select statement (the `k` argument to
                this function) is passed in here.
            * table_info: Table definitions and sample rows are passed in here. If the
                user specifies "table_names_to_use" when invoking chain, only those
                will be included. Otherwise, all tables are included.
            * dialect (optional): If dialect input variable is in prompt, the db
                dialect will be passed in here.

        Here's an example prompt:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate

            template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Use the following format:

            Question: "Question here"
            SQLQuery: "SQL Query to run"
            SQLResult: "Result of the SQLQuery"
            Answer: "Final answer here"

            Only use the following tables:

            {table_info}.

            Question: {input}'''
            prompt = PromptTemplate.from_template(template)
    """  # noqa: E501
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables)
    ):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )


top_k=10

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain_v2(llm, db, k=top_k)
agentSQL = write_query | execute_query

def agent_SQL(state: State):
    messages=state["messages"]
    response=agentSQL.invoke({"question": messages[-1].content})
    return {"messages": [response]}

#while True:
 #   user_input=(input("user input:"))
#
  #  print(agentSQL.invoke({"question": "Find cars that match the following criteria:\n - color: blue"}))
  #  print(agentSQL.invoke({"question": user_input})) 