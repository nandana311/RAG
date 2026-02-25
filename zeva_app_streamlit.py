import os
import uuid
from dotenv import load_dotenv
import streamlit as st

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FIX: Match this to your ingestion script folder name!
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_chat") 
load_dotenv(os.path.join(BASE_DIR, ".env"))

# --- LangGraph RAG Backend ---
class RAGState(MessagesState):
    context: str

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    collection_name="demo",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

llm = init_chat_model(model="gpt-4o-mini", temperature=0, model_provider="openai")


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Zeva, a specialized Carbon Compliance & Sustainability Analyst. "
        "Your goal is to help businesses navigate the complex 2026 carbon regulations. "
        "Rely strictly on the provided context to find emission data, regulatory deadlines, and compliance gaps. "
        "If the data is not in the context, state exactly what is missing rather than guessing. "
        "Use a professional, diligent, and data-driven tone."
    ),
    MessagesPlaceholder(variable_name="history", n_messages=3),
    (
        "system", 
        "Context (Regulatory & Corporate Data):\n{context}"
    ),
    MessagesPlaceholder(variable_name="question", n_messages=1),
])
chain = prompt | llm

def retriever_node(state: RAGState):
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"context": "\n\n".join(d.page_content for d in docs)}

def chat_node(state: RAGState):
    ai = chain.invoke({
        "history": state["messages"][:-1],
        "question": state["messages"][-1:],
        "context": state.get("context", "")
    })
    return {"messages": [ai]}

builder = StateGraph(RAGState)
builder.add_node("retriever_node", retriever_node)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "retriever_node")
builder.add_edge("retriever_node", "chat_node")
builder.add_edge("chat_node", END)
graph = builder.compile(checkpointer=MemorySaver())

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Zeva RAG", layout="wide")

# Updated CSS for clean bubbles and centering
st.markdown("""
    <style>
    .chat-row { display: flex; margin-bottom: 15px; width: 100%; }
    .user-row { justify-content: flex-end; }
    .bot-row { justify-content: flex-start; }
    
    .bubble {
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 75%;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .user-bubble {
        background-color: #0d6efd;
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bot-bubble {
        background-color: #f0f2f6;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 4px;
    }
    /* This centers the conversation like ChatGPT */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# st.markdown("<h2 style='text-align: center;'>Zeva Says Hi!</h2>", unsafe_allow_html=True)

# --- Welcome message logic ---
WELCOME_MSG = (
    "Hi! I am Zeva, your Carbon Compliance Analyst. "
    "I have been trained on the latest EU CBAM 2026 regulations and corporate ESG disclosures. "
    "Ask me to identify compliance gaps, calculate emission intensities, or summarize sustainability targets from your documents."
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"st-chat-{uuid.uuid4().hex[:8]}"
if "history" not in st.session_state or not st.session_state.history:
    st.session_state.history = [("Zeva", WELCOME_MSG)]


# 1. Display Chat History
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f'<div class="chat-row user-row"><div class="bubble user-bubble"><b>You</b><br>{text}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-row bot-row"><div class="bubble bot-bubble"><b>Zeva</b><br>{text}</div></div>', unsafe_allow_html=True)

# 2. Native Fixed Bottom Input
# This component is ALWAYS fixed to the bottom by default in Streamlit
user_input = st.chat_input("Ask your question here...")

if user_input:
    # Append user message to UI
    st.session_state.history.append(("You", user_input))
    
    # Trigger Rerender to show User message immediately
    st.rerun()

# 3. Handle the Logic (After rerun, if last message is from 'You')
if st.session_state.history and st.session_state.history[-1][0] == "You":
    latest_query = st.session_state.history[-1][1]
    
    with st.spinner("Thinking..."):
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=latest_query)]},
                {"configurable": {"thread_id": st.session_state.thread_id}}
            )
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"Error: {e}"
            
    st.session_state.history.append(("Zeva", answer))
    st.rerun()

# 4. Sidebar for "New Chat" and Stats
with st.sidebar:
    st.markdown("# Zeva Assistant")
    st.markdown("---")
    if st.button("New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.history = []
        st.rerun()