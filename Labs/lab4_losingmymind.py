import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

# A fix for working with ChromaDB on Streamlit Community Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('Lab4Collection')

#### USING CHROMA DB WITH OPENAI EMBEDDINGS ####

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# A function that will add documents to collection
# collection = ChromaDB collection, already established
# text = extracted text from PDF files
# Embeddings inserted into the collection from OpenAI
def add_to_collection(collection, text, file_name):
    # Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    
    # Get the embedding
    embedding = response.data[0].embedding
    
    # Add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{"filename": file_name}]
    )

#### EXTRACT TEXT FROM PDF ####
# This function extracts text from each syllabus
# to pass to add_to_collection
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
        return ""

#### POPULATE COLLECTION WITH PDFs ####
# This function uses extract_text_from_pdf
# and add_to_collection to put syllabi in ChromaDB collection
def load_pdfs_to_collection(folder_path, collection):
    pdf_folder = Path(folder_path)
    if not pdf_folder.exists():
        st.warning(f"Folder '{folder_path}' not found.")
        return 0
    
    pdf_files = list(pdf_folder.glob('*.pdf'))
    if not pdf_files:
        st.warning(f"No PDF files found in '{folder_path}'.")
        return 0
    
    loaded = 0
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            add_to_collection(collection, text, pdf_file.name)
            loaded += 1
    
    return loaded

# Check if collection is empty and load PDFs
if collection.count() == 0:
    with st.spinner("Loading PDFs into ChromaDB..."):
        loaded = load_pdfs_to_collection('./lab4pdfs', collection)
        st.success(f"✅ Loaded {loaded} PDF documents into ChromaDB")

#### MAIN APP ####
st.title('Lab 4: Chatbot using RAG')

#### QUERYING A COLLECTION — ONLY USED FOR TESTING ####

topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

if topic:
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=topic,
        model='text-embedding-3-small'
    )
    
    # Get the embedding
    query_embedding = response.data[0].embedding
    
    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # The number of closest documents to return
    )
    
    # Display the results
    st.subheader(f'Results for: {topic}')
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        
        st.write(f'**{i+1}. {doc_id}**')

else:
    st.info('Enter a topic in the sidebar to search the collection')

#### SIDEBAR INFO ####
st.sidebar.divider()
st.sidebar.header("📊 Database Info")
st.sidebar.write(f"**Collection:** Lab4Collection")
st.sidebar.write(f"**Documents loaded:** {collection.count()}")