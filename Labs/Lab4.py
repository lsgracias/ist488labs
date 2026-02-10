__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import os

st.title("📚 Lab 4 - RAG Course Information Chatbot")
st.write("""
This chatbot uses **RAG (Retrieval-Augmented Generation)** to answer questions about course syllabi.
It searches through course PDF documents and provides answers based on the relevant content found.
Courses: IST 195, IST 256, IST 314, IST 343, IST 387, IST 418, IST 488
""")

# Get API key from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Configuration
COLLECTION_NAME = "Lab4Collection"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
PDF_FOLDER = "lab4pdfs"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text.strip()
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
        return ""

# Function to create ChromaDB collection
def create_vector_db():
    # Create OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL
    )
    
    # Create ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    
    # Check if PDFs folder exists
    if not os.path.exists(PDF_FOLDER):
        st.warning(f"PDF folder '{PDF_FOLDER}' not found. Please create it and add PDF files.")
        return collection
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.warning(f"No PDF files found in '{PDF_FOLDER}' folder.")
        return collection
    
    # Process each PDF
    documents = []
    metadatas = []
    ids = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            documents.append(text)
            metadatas.append({"filename": pdf_file, "source": pdf_path})
            ids.append(pdf_file)  # Use filename as unique ID
    
    # Add documents to collection
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.sidebar.success(f"✅ Loaded {len(documents)} PDF documents")
    
    return collection

# Function to query vector database
def query_vector_db(collection, query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

# Function to format context from query results
def format_context(results):
    context = ""
    if results and results['documents'] and results['documents'][0]:
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            filename = metadata.get('filename', 'Unknown')
            # Truncate document to avoid token limits
            doc_truncated = doc[:3000] if len(doc) > 3000 else doc
            context += f"\n\n--- Document {i+1}: {filename} ---\n{doc_truncated}"
    return context

# Function to get LLM response
def get_llm_response(query, context, chat_history):
    system_prompt = f"""You are a helpful course information assistant. You answer questions based on the course syllabus documents provided.
IMPORTANT RULES:
1. Answer questions using ONLY the information from the provided course documents.
2. If you find relevant information, clearly state which course/document it comes from.
3. If the answer is not in the provided documents, say "I couldn't find that information in the course documents."
4. Be helpful, clear, and concise.
5. When using information from the documents, mention that you found it in the course materials.

Here are the relevant course documents:
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (last 8 messages)
    for msg in chat_history[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    )
    return stream

# --- Initialize Vector Database ---
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Creating vector database from PDF documents..."):
        st.session_state.Lab4_VectorDB = create_vector_db()

# Initialize chat history
if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = []

# --- Sidebar Info ---
st.sidebar.header("RAG Info")
st.sidebar.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
st.sidebar.write(f"**LLM Model:** {LLM_MODEL}")
st.sidebar.write(f"**Collection:** {COLLECTION_NAME}")

# Show loaded documents
if st.session_state.Lab4_VectorDB:
    try:
        count = st.session_state.Lab4_VectorDB.count()
        st.sidebar.write(f"**Documents loaded:** {count}")
    except:
        st.sidebar.write("**Documents loaded:** 0")

st.sidebar.divider()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.lab4_messages = []
    st.rerun()

# --- Main Chat Interface ---

# Display chat history
for message in st.session_state.lab4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the courses..."):
    # Add user message to history
    st.session_state.lab4_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Query vector database for relevant documents
    with st.spinner("Searching course documents..."):
        results = query_vector_db(st.session_state.Lab4_VectorDB, prompt, n_results=3)
        context = format_context(results)
    
    # Show which documents were retrieved (in expander)
    if results and results['metadatas'] and results['metadatas'][0]:
        with st.expander("📄 Documents used for this response"):
            for i, metadata in enumerate(results['metadatas'][0], 1):
                st.write(f"{i}. {metadata.get('filename', 'Unknown')}")
    
    # Generate response
    with st.chat_message("assistant"):
        if context:
            stream = get_llm_response(prompt, context, st.session_state.lab4_messages)
            response = st.write_stream(stream)
        else:
            response = "I don't have any course documents loaded. Please make sure PDF files are in the 'pdfs' folder."
            st.write(response)
    
    # Add assistant response to history
    st.session_state.lab4_messages.append({"role": "assistant", "content": response})