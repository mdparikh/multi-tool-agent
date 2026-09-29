from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceHub
from pathlib import Path
import sqlite3
import os
import re

from dotenv import load_dotenv

load_dotenv()

# ------------------------- RAG tool (Persistent Chroma DB)
SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "pdfs"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"
DB_PATH = SCRIPT_DIR / "company.db"

embeddings = FakeEmbeddings(size=384)

if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
    # Fast path: Load already-indexed vector store from disk
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    # Slow path: Index once and save to disk
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    raw_documents = loader.load()

    if raw_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(raw_documents)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        retriever = None

@tool
def company_rag_search(query: str) -> str:
    """Provides information from the landmark paper on Transformers 'Attention is all you need'"""
    if retriever is None:
        return "No PDF documents are currently loaded in the database."
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in documents."
    return "\n\n".join([f"Source: {d.metadata.get('source', 'PDF')}\nContent: {d.page_content}" for d in results])


# ---------------------------- SQL Tool (Initialize once)

needs_init = not DB_PATH.exists()
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
if needs_init:
    cursor.execute("CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT)")
    cursor.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000)")
    cursor.execute("INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 75000)")
    cursor.execute("INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 105000)")
    conn.commit()
conn.close()

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

@tool
def query_database(query: str) -> str:
    """Executes a SQL query against the company database and returns results."""
    try:
        return str(db.run(query))
    except Exception as e:
        return f"Database query error: {str(e)}"

# -----------------Create Search tool
web_search=TavilySearchResults(max_results=2)

tools=[query_database,company_rag_search,web_search]

# ------------------- LLM Router

def get_llm(provider:str):
    p=provider.lower()
    if p=="gemini": return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    if p=="groq": return ChatGroq(model_name="openai/gpt-oss-120b",temperature=0)
    if p=="huggingface": return HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",model_kwargs={"temperature":0.1,"max_new_tokens":512})
    raise ValueError("Unsupported")
# print("LLM Router configured.")

# -----------------------Input and Output Guardrails


BLOCKED_KEYWORDS = ["ignore previous instructions", "drop table", "system prompt", "hack", "bypass"]

def apply_input_guardrails(prompt: str) -> tuple[bool, str]:
    """Validates user input against safety guidelines."""
    # Check 1: Prompt Injection Attacks
    for keyword in BLOCKED_KEYWORDS:
        if keyword in prompt.lower():
            return False, f"Flagged: Potential prompt injection or malicious query detected ('{keyword}')."

    # Check 2: Prompt Length Validation
    if len(prompt.strip()) < 3:
        return False, "Flagged: Query is too short."

    return True, "Passed Input Guardrails"

def apply_output_guardrails(response_text: str) -> str:
    """Validates and sanitizes model output before returning to user."""
    # Mask PII / Sensitive Patterns (e.g., Credit Cards / SSNs)
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    sanitized_text = re.sub(ssn_pattern, "[REDACTED PII]", response_text)

    # Simple toxic/unsafe output check fallback
    if "UNSAFE_OUTPUT_FLAG" in sanitized_text:
        return "Internal Safety Warning: Output generated violated response guidelines."

    return sanitized_text

# print("Guardrails initialized.")

# ------------------- Agentic Flow with Langfuse Tracing

# Create Langfuse handler for telemetry tracing
langfuse_handler = CallbackHandler()
parser = StrOutputParser()

def run_agent_pipeline(user_prompt: str, provider: str = "gemini") -> dict:

    # Step 1: Input Guardrail Check
    is_safe, guardrail_msg = apply_input_guardrails(user_prompt)
    if not is_safe:
        return {
            "status": "blocked_by_input_guardrail",
            "response": guardrail_msg,
            "provider": provider
        }

    # Step 2: Initialize Chosen LLM
    try:
        llm = get_llm(provider)
    except Exception as e:
        return {"status": "error", "response": f"Model Provider Error: {str(e)}"}

    # Step 3: Create ReAct Agent with Tools

    agent = create_agent(model=llm, tools=tools)

    # Step 4: Synchronous Execution with Langfuse Telemetry Tracing
    try:
        inputs = {"messages": [("user", user_prompt)]}
        result = agent.invoke(
            inputs,
            config={"callbacks": [langfuse_handler]}
        )
        
        # Pass the final AIMessage into StrOutputParser
        final_message = result["messages"][-1]
        raw_output = parser.invoke(final_message)

        # Step 5: Output Guardrail Verification
        sanitized_output = apply_output_guardrails(raw_output)

        return {
            "status": "success",
            "response": sanitized_output,
            "provider": provider
        }
    except Exception as e:
        return {"status": "error", "response": f"Agent Execution Failed: {str(e)}"}

# print("Agent Execution Flow Ready (using create_agent).")
