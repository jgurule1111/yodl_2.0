import streamlit as st
import os
import textwrap
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
import time
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain.schema import HumanMessage, SystemMessage 
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from transformers import pipeline
import time
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pprint import pprint
import uuid



os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "langchain rag agent"

os.environ['GROQ_API_KEY'] = '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'




@tool
def calculate_growth_rate(present_value, past_value):
    """
    Calculate the growth rate given the initial and final values.

    Parameters:
    - present_value (float): The final value.
    - past_value (float): The initial value.

    Returns:
    - float: The growth rate as a percentage.

    Example:
    calculate_growth_rate(150, 100) -> 50.0
    """
    try:
        growth_rate = ((present_value - past_value) / past_value) * 100
        round_growth_rate = round(growth_rate, 2)
        return round_growth_rate
    except ZeroDivisionError:
        return print("error, Initial value cannot be zero")

@tool
def calculate_quick_ratio(current_assets, inventory, current_liabilities):
    """
    Calculate the net working capital.

    Parameters:
    - current_assets (float): Total current assets of the company.
    - long_term_debt (float): Total long-term debt of the company.
    - current_liabilities (float): Total current liabilities of the company.

    Returns:
    - float: The net working capital.

    Example:
    net_working_capital(100000, 40000, 30000) -> 30000
    """
    try:
        quick_ratio = (current_assets - inventory) / current_liabilities
        return quick_ratio
    except ZeroDivisionError:
        return print("error Current liabilities cannot be zero")

@tool
def net_working_capital(current_assets, long_term_debt, current_liabilities):
  """    Calculate the net working capital.

    Parameters:
    - current_assets (float): Total current assets of the company.
    - long_term_debt (float): Total long-term debt of the company.
    - current_liabilities (float): Total current liabilities of the company.

    Returns:
    - float: The net working capital.

    Example:
    net_working_capital(100000, 40000, 30000) -> 30000"""
  try:
    net_working_capital = current_assets - (long_term_debt + current_liabilities)
    return net_working_capital
  except ZeroDivisionError:
    return print("error Current liabilities cannot be zero")


@tool
def generate_graph(name1:str, name2:str, name3:str, ticker="NKE"):
  """
    Generate a line graph to visualize financial data (e.g., EPS, revenue, and free cash flow) over time for a specified company.

    Parameters:
    - name1 (str): The name of the first financial metric (e.g., 'EPS').
    - name2 (str): The name of the second financial metric (e.g., 'Revenue').
    - name3 (str): The name of the third financial metric (e.g., 'Free Cash Flow').
    - ticker (str): The stock ticker symbol of the company (default is 'NKE').

    Returns:
    - Matplotlib figure: A line graph showing the financial metrics over time.

    Example:
    generate_graph('EPS', 'Revenue', 'Free Cash Flow', 'NKE')
    """
  company_name = yf.Ticker(ticker)
  income_statement = company_name.income_stmt

  df = pd.DataFrame(income_statement)

  random_name1 = df.loc[name1]
  random_name2 = df.loc[name2]
  random_name3 = df.loc[name3]

  #cash_flow = company_name.cashflow
  #df = pd.DataFrame(cash_flow)

  #fcf = df.loc[name3]

  # Plot the data
  #plt.figure(figsize=(10, 6))

  plt.plot(random_name1, marker='o', label=name1)
  plt.plot(random_name2, marker='o', label=name2)
  plt.plot(random_name3, marker='o', label=name3)
  plt.legend()
  plt.title('Growth over the last few years')
  plt.xlabel('Date')
  plt.ylabel('($) in billions')
  plt.xticks(rotation=45)
  plt.tight_layout()
  #plt.ylim([0, 50])
  return plt.show()

@tool
def calculate_cagr(present_value, past_value, time):
    """Calculates the Compound Annual Growth Rate (CAGR) given the present value, past value, and the number of years.

    Parameters:
    - present_value (float): The value at the end of the period.
    - past_value (float): The value at the beginning of the period.
    - time (float): The number of years over which the growth is measured.

    Returns:
    - dict: A dictionary containing either the calculated CAGR (rounded to two decimal places) under the key "messages" or an error message under the key "error" if the calculation could not be performed.

    Note:
    - If the past value is zero, the function will return an error message indicating that the initial value cannot be zero.
    - Any other exceptions encountered during the calculation will be returned as error messages.
    """
    try:
        if past_value == 0:
            return print({"error": "Initial value cannot be zero"})
        cagr = ((present_value / past_value) ** (1 / time) - 1) * 100
        cagr_round = round(cagr, 2)
        return cagr_round #print({"messages": cagr_round})
        #round(calculation, 2)
    except Exception as e:
        return print("error")

#!mkdir data
#!wget "https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf" -O "data/Nike_annual_23.pdf"


instruction = """The provided document is Nikes 2023 Annual Report Results.
This form provides detailed financial information about the company's performance for a specific year.
It includes the audited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
It also contains many tables.
Try to be precise while answering the questions, if you don't know the answer, say you don't know, don't try to make up an answer."""

parser = LlamaParse(
    api_key="llx-LTFMoRgHKnsJqQuGkL9rmIn4QzpL9ESXmM0MTZ7FsmQ0VPpy",
    result_type="markdown",
    parsing_instruction=instruction,
    max_timeout=5000,
)



llama_parse_documents = parser.aload_data("Nike2023.pdf")
parsed_doc = llama_parse_documents

document_path = Path("parsed_document.md")
#with document_path.open("a") as f:
    #f.write(parsed_doc.text)

loader = UnstructuredMarkdownLoader(document_path)
loaded_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4048, chunk_overlap=128)
doc = text_splitter.split_documents(loaded_documents)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

qdrant = Qdrant.from_documents(
    doc,
    embeddings,
    # location=":memory:",
    path="db",
    collection_name="document_embeddings",
)

#A retriever is an interface that returns documents given an unstructured query.
retriever = qdrant.as_retriever(search_kwargs={"k": 3})


compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


tools = [
    net_working_capital,
    calculate_quick_ratio,
    calculate_growth_rate,
    generate_graph,
    calculate_cagr
]
def income_statement(ticker: str, question: str):
  """
    This function fetches the income statement of a given company using its stock ticker symbol,
    converts the income statement into a DataFrame, and uses a language model to analyze the data
    and answer a specified question.

    Parameters:
    ticker (str): The stock ticker symbol of the company.
    question (str): The question you want to ask about the company's income statement."""

  company_name = yf.Ticker(ticker)
  income_statement = company_name.income_stmt

  df = pd.DataFrame(income_statement)

  df_str = df.to_string()

  documents = df_str

  questions = [HumanMessage(content=question)]



  return {"documents": str(df_str)}


def balance_sheet(ticker: str, question: str):
  """
    This function fetches the income statement of a given company using its stock ticker symbol,
    converts the income statement into a DataFrame, and uses a language model to analyze the data
    and answer a specified question.

    Parameters:
    ticker (str): The stock ticker symbol of the company.
    question (str): The question you want to ask about the company's income statement."""

  company_name = yf.Ticker(ticker)
  balance_sheet = company_name.balance_sheet

  df = pd.DataFrame(balance_sheet)

  df_str = df.to_string()

  documents = df_str

  questions = [HumanMessage(content=question)]

  return {"documents": str(df_str)}




class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    messages : Annotated[list[AnyMessage], add_messages]
    #document: str
    question : str
    #generation : str
    documents : List[Document]


#############################TEST
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: GraphState, config: RunnableConfig):
      while True:





        result = self.runnable.invoke(state)

        #messages = state["messages"]
        #response = llm.invoke(result)


        #append = state['messages'].append(state['question'])
        del state['question']


        #result is now generation = chain_mainy.invoke({"documents": documents, "question": question})




        #tool_calls
        if not result.tool_calls and (
            not result.content
            or isinstance(result.content, list)
            and not result.content[0].get("text")
        ): #then retry

           messages = state["messages"] + [("user", "Respond with a real output")] #+ state["documents"]
           state = {**state, "messages": result}
        else:
          break


      return {"messages": [result]}





def retrieve(state):
  """
  Retrieve documents from vectorstore

  Args:
      state (dict): The current graph state

  Returns:
      state (dict): New key added to state, documents, that contains retrieved documents
  """


  #question = state["question"]

  #In order to remove an element from the beginning of a tuple, we will make a new tuple with the remaining elements as shown below
  question_myTuple = state["question"]     #this is where the question goes
  question_myTuple = question_myTuple[0:]  #this takes out the first word in the question which is 'user'
  final_tuple = " ".join(question_myTuple)#this turns the tuple into a string

  balance = []

  if "calculate" in question_myTuple:
    yfinance_result = balance_sheet("NKE", question_myTuple)
    return {"documents": yfinance_result, "messages": question_myTuple}


  else:
    documents = compression_retriever.invoke(question_myTuple) #later I should add a chain/prompt to make this better
    return {"documents": documents, "messages": question_myTuple}#{"documents": documents, "question": question_myTuple}


def generate(state):
  """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
  #question = state["question"]

  append = state['messages'].append(state['question'])
  del state['question']
  messages = state["messages"]
  documents = state["documents"]

    # RAG generation [llm_with_tools.invoke(state["question"])]

  generation = assistant_runnable.invoke({"documents": documents, "messages": messages})

  list_gen = state["messages"]

  list_gen.append(generation)
  stater = {**state, "messages": list_gen}

  regenerate = assistant_runnable.invoke({"documents": documents, "messages": stater})
  return print(regenerate)



assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (   "system",
            '''Please use the retrieved documents to answer the user's question thoroughly.

- For questions involving calculations, use the following list of tools to provide the best answer. DO NOT CALCULATE THE ANSWER YOURSELF.
- List of tools:
  - calculate growth rates
  - calculate net working capital
  - calculate the quick ratio
  - calculate the CAGR
- For questions requiring graphs, use the generate_graph function.

Ensure your response is accurate and relevant by effectively incorporating the provided documents.

Retrieved Documents: {documents}''',

        ),
        ("placeholder", "{question}"),
    ]
)


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (   "system",
            """Please use the retrieved documents to answer the user's question thoroughly.

- For questions involving financial calculations, use the appropriate tool from the list below:
  - For growth rates, use the calculate_growth_rate tool.
  - For net working capital, use the net_working_capital tool.
  - For quick ratio, use the calculate_quick_ratio tool.
  - For compound annual growth rate (CAGR), use the calculate_cagr tool.

- For questions requiring visual representation of data, use the generate_graph function.

Do not perform any calculations yourself. Ensure your response is accurate and relevant by effectively incorporating the provided documents.

Retrieved Documents: {documents},"""

        ),
        ("placeholder", "{messages}"),
    ]
)



'''primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (   "system",
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved documents to answer the question, utilizing the tools when necessary.
            You also have access to three tools if needed to answers the users question:

            (1) calculating growth rates,
            (2) calculating net working capital,
            (3) calculating the quick ratio.

Some questions will explicitly require the use of one of these tools,
while other questions may not involve tool use at all. For questions that do not require tool use, you are expected to generate answers from the retrieved documents.

Here are some guidelines to follow:

- If the user explicitly asks you to use one of the tools, employ the appropriate tool to provide the answer.
- If the user's question does not involve any of the tools, generate a well-informed and accurate response based on the documents.
- If you do not know the answer to a question, be honest and say "I don't know the answer."
- Ensure that your answers are clear, concise, and helpful, regardless of whether tools are used or not.

Remember, your goal is to provide valuable assistance to the users by leveraging your tools when applicable and relying on the retrieved documents when necessary.
Retrieved Documents relevant to the question: {documents}""",

        ),
        ("placeholder", "{messages}"),
    ]
)

'''
from langchain_openai import ChatOpenAI

#model = ChatOpenAI(model="gpt-3.5-turbo")
# LLM chain
llm = ChatGroq(temperature=0, model="llama3-groq-70b-8192-tool-use-preview")
from langchain.output_parsers import PandasDataFrameOutputParser
parser = PandasDataFrameOutputParser()
# Modification: bind_tools: tell the LLM which tools it can call
assistant_runnable = primary_assistant_prompt | model.bind_tools(tools)

assistant_runnable123 = assistant_prompt | llm


from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=5000):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

####TESTERR

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

builder = StateGraph(GraphState)




builder.set_entry_point("retrieve")
builder.add_node("retrieve", retrieve)
#builder.add_node("generate", generate)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
#builder.add_node("tools", tool_node)
builder.add_edge("retrieve", "assistant")
builder.add_edge("tools", "assistant")

builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
    {"tools": "tools", END: END},
)


def test_poop(question:str):
  _printed = set()
  thread_id = str(uuid.uuid4())

  config = {
      "configurable": {
          # Checkpoints are accessed by thread_id
          "thread_id": thread_id,
      }
  }

  events = graph.stream(
      {"question": question}, config, stream_mode="values"

  )

  for event in events:
    _print_event(event, _printed)
  
  
