import streamlit as st
import os
import shutil
import uuid
from dotenv import load_dotenv
from gtts import gTTS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

import asyncio
from langchain.chat_models import ChatOpenAI



# Load environment variables
load_dotenv()

FAISS_INDEX_PATH = "./faiss_index"
AUDIO_PATH = "./audio_responses"  # Directory for storing audio responses

# Ensure directories exist
os.makedirs(AUDIO_PATH, exist_ok=True)










# ✅ Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello there !!. Please enter a Web URL for starting the chat.")]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "website_url" not in st.session_state:
    st.session_state.website_url = None

if "audio_files" not in st.session_state:
    st.session_state.audio_files = []  # Store generated audio file paths
if "summary" not in st.session_state:
    st.session_state.summary = None

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None      








# ✅ Function to delete only session-tracked audio files
def delete_audio_files():
    for file_path in st.session_state.audio_files:
        if os.path.isfile(file_path):
            os.remove(file_path)
    st.session_state.audio_files = []  # Clear the list after deletion

# ✅ Function to create FAISS vector store from a website URL


async def generate_summary_async(document_chunks):
    llm = ChatOpenAI()  # Use OpenAI API efficiently
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Process chunks asynchronously for faster execution
    summary = await summarize_chain.ainvoke(document_chunks)
    
    return summary["output_text"].strip()

# Run the async function
def generate_summary(document_chunks):
    return asyncio.run(generate_summary_async(document_chunks))



# Function to create FAISS vector store from a website URL
def get_vectorstore_from_url(url):
    try:
        shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        loader = WebBaseLoader(url)
        document = loader.load()
        if not document:
            st.error("No content found on the provided URL.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.document_chunks = text_splitter.split_documents(document)

        vector_store = FAISS.from_documents(
            documents=st.session_state.document_chunks,
            embedding=OpenAIEmbeddings()
        )
        vector_store.save_local(FAISS_INDEX_PATH)
       

        # Generate summary
        
        ##st.session_state.chat_history.append(AIMessage(content=f"Website Summary: {summary}"))

        return vector_store
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        return None


# ✅ Function to create a context-aware retriever
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Use the above conversation history to generate a search query and fetch the most relevant context."),
    ])

    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# ✅ Function to generate unique speech file
def generate_speech(text):
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    filename = f"response_{unique_id}.mp3"  # Unique filename
    filepath = os.path.join(AUDIO_PATH, filename)

    try:
        # Generate TTS file
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)
        st.session_state.audio_files.append(filepath)  # Store in session state
        return filepath
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None
    








# ✅ Function to get AI response
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']




st.markdown(
    """
    <style>
        .title-container {
            text-align: center;
            padding: 10px;
            margin-bottom: 20px; /* Added space below the title */
        }
        .title {
            font-size: 36px;  /* Slightly increased font size */
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            padding: 8px;
            border-bottom: 2px solid white;
            box-shadow: 0px 2px 6px rgba(255, 255, 255, 0.2);
            display: inline-block;
        }
    </style>
    <div class="title-container">
        <span class="title">WEBWHISPER - Talk to Websites!</span>
    </div>
    """,
    unsafe_allow_html=True
)






with st.sidebar:
    website_url = st.text_input("🔗 Enter Website URL")

if website_url:
    # Delete only session audio files when the website URL changes
    if st.session_state.website_url != website_url:
        st.session_state.website_url = website_url
        delete_audio_files()  # Delete only session-tracked audio files
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        if st.session_state.vector_store:
           st.session_state.chat_history = [
    AIMessage(content="Fetched Data from the Website URL successfully ! . Please feel free to ask any queries regarding this Website!\n")
] 
        
    if st.button("Generate Summary"):
            st.session_state.summary=generate_summary(st.session_state.document_chunks)
            st.session_state.chat_history.append(AIMessage(content="SUMMARY:\n" + st.session_state.summary))
           


# ✅ Chat input and processing
user_query = st.chat_input("💬 Type your message here...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Cant fetch Data . Please enter a valid URL first.")
    else:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# ✅ Display chat messages with "Play" button for AI responses
for index, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)

            # 🎯 "Play" button to generate and play TTS only when clicked
            if st.button(f"▶️ listen", key=f"play_{index}"):
                delete_audio_files()
                audio_file = generate_speech(message.content)  # Generate audio only on button click
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                else:
                    st.error("Failed to generate audio.")

    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)