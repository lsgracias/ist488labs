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

# Helper function for chunking text
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# Function to create ChromaDB collection
def create_vector_db():
    pdf_dir = "./lab4pdfs"
    
    # Check if directory exists
    if not os.path.exists(pdf_dir):
        st.error(f"Directory {pdf_dir} not found.")
        return None

    # Retrieve all PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.error(f"No PDF files found in {pdf_dir}.")
        return None
        
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create or get collection
    # We use a new name to force a fresh start if the code changes
    collection_name = "Lab4Collection_v2" 
    
    try:
        # Try to delete if it exists to ensure freshness (optional, but good for dev)
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    
    documents = []
    metadatas = []
    ids = []

    id_counter = 0
    
    # Process each PDF
    for filename in pdf_files:
        file_path = os.path.join(pdf_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text()
                
                # Chunk the text
                chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    ids.append(f"{filename}_chunk_{i}") # Unique ID per chunk
                    metadatas.append({"filename": filename, "chunk_id": i})
                    
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    # Generate embeddings using batches to avoid API limits
    embeddings = []
    batch_size = 100
    
    progress_bar = st.progress(0, text="Generating embeddings...")
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch_docs, model="text-embedding-3-small")
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            progress_bar.progress((i + len(batch_docs)) / len(documents), text=f"Generated {i + len(batch_docs)}/{len(documents)} embeddings")
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}: {e}")
            return None # Stop if embedding fails
            
    progress_bar.empty()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

# --- Initialize Vector Database ---
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Creating vector database from PDF documents..."):
        st.session_state.Lab4_VectorDB = create_vector_db()

# Initialize chat history
if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = []

# --- Sidebar Info ---
st.sidebar.header("RAG Info")
st.sidebar.write(f"**Embedding Model:** text-embedding-3-small")
st.sidebar.write(f"**LLM Model:** gpt-4o-mini")

# Show loaded documents
if st.session_state.Lab4_VectorDB:
    try:
        count = st.session_state.Lab4_VectorDB.count()
        st.sidebar.write(f"**Documents loaded:** {count}")
    except:
        st.sidebar.write("**Documents loaded:** 0")

st.sidebar.divider()

# Clear chat button
if st.sidebar.button("Clear Chat"):
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
    
    # Retrieve relevant documents
    context_text = ""
    retrieved_docs = []

    if st.session_state.Lab4_VectorDB:
        with st.spinner("Searching course documents..."):
            query_response = client.embeddings.create(input=prompt, model="text-embedding-3-small")
            query_embedding = query_response.data[0].embedding
            
            results = st.session_state.Lab4_VectorDB.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    filename = results['metadatas'][0][i]['filename']
                    context_text += f"\n\n--- Document Snippet {i+1} (Source: {filename}) ---\n{doc}"
                    retrieved_docs.append(filename)

    system_prompt = f"""You are a helpful course information assistant. You answer questions based on the course syllabus documents provided.
IMPORTANT RULES:
1. If you find relevant information, clearly state which course/document it comes from.
2. If the answer is not in the provided documents, say "I couldn't find that information in the course documents."
3. Be helpful, clear, and concise.
4. When using information from the documents, mention that you found it in the course materials.

Here are the relevant course documents:
{context_text}
"""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add current query
    messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Add assistant response to history
    st.session_state.lab4_messages.append({"role": "assistant", "content": response})