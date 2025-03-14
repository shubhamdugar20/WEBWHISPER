import streamlit as st 
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")  

import os
import shutil  
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables
load_dotenv()

VECTOR_DB_PATH = "./chroma_db"

# ✅ Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "website_url" not in st.session_state:
    st.session_state.website_url = None


# Function to create Chroma vectorstore from a website URL
def get_vectorstore_from_url(url):
    """Scrapes the given URL, extracts content, and creates a Chroma vector store."""
    try:
        shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)

        loader = WebBaseLoader(url)
        document = loader.load()
        if not document:
            st.error("No content found on the provided URL.")
            return None

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        document_chunks = text_splitter.split_documents(document)

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=VECTOR_DB_PATH
        )
        vector_store.persist()  # Save vector store
        st.success("Website data successfully indexed!")

        return vector_store

    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        return None


# Function to create a context-aware retriever
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Use the above conversation history to generate a search query and fetch the most relevant context."),
    ])

    return create_history_aware_retriever(llm, retriever, prompt)


# Function to create a conversational RAG chain
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# Function to get chatbot response
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


# Streamlit App Configuration
st.title("Chat with Websites")

# Sidebar - Enter Website URL
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url:
    if st.session_state.website_url != website_url:  # Prevent unnecessary reprocessing
        st.session_state.website_url = website_url
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        if st.session_state.vector_store:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# Chat input and processing
user_query = st.chat_input("Type your message here...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Vector store not initialized. Please enter a valid URL first.")
    else:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# Display chat messages
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)
