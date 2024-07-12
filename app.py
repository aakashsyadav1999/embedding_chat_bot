import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
import pinecone
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = "3c3dfb50-66f5-4212-bd6e-f26b5f964b8d"
PINECONE_API_HOST = "https://aiagent-axzviig.svc.aped-4627-b74a.pinecone.io"
index_name = "aiagent"

# Function to load PDF files
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Function to download Hugging Face embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Function to search the index with a query
def search_index(query, model, index, text_chunks):
    query_embedding = model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=5, include_values=True, include_metadata=True)
    return result


pc = Pinecone(
                api_key=PINECONE_API_KEY,
                host=PINECONE_API_HOST,
                environment="us-east-1" 
)

#index name
index_name = "aiagent"

# Connect to the index
index = pc.Index(index_name,host=PINECONE_API_HOST)

#Check if the index exists and create it if it doesn't


# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = download_hugging_face_embeddings()

# Streamlit UI
st.title("PDF Query App")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your query")

if uploaded_files and query:
    # Save uploaded files to a temporary directory
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Load and process PDFs
    documents = load_pdf(temp_dir)
    text_chunks = text_split(documents)
    
    # Embed text chunks
    text_embeddings = [embeddings.embed_query(t.page_content) for t in text_chunks]
    
    # Upsert embeddings into Pinecone
    items = [(str(i), embedding) for i, embedding in enumerate(text_embeddings)]
    index.upsert(items)
    
    # Search the index with the query
    result = search_index(query, model, index, text_chunks)
    
    # Display the top result
    if result['matches']:
        top_match = result['matches'][0]
        st.write("Top Result:")
        st.write(f"Score: {top_match['score']}, Content: {text_chunks[int(top_match['id'])].page_content}")

    # Clean up temporary directory
    for uploaded_file in uploaded_files:
        os.remove(os.path.join(temp_dir, uploaded_file.name))

else:
    st.write("Please upload PDF files and enter a query.")