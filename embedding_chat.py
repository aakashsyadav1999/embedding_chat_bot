from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import pinecone
from pinecone import Pinecone, ServerlessSpec
import time
from sentence_transformers import SentenceTransformer
import os

PINECONE_API_KEY = "3c3dfb50-66f5-4212-bd6e-f26b5f964b8d"
PINECONE_API_HOST = "https://aiagent-axzviig.svc.aped-4627-b74a.pinecone.io"

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Path to the directory containing PDF files
extracted_data = r'D:\VS code files\building_energy_consumption\pdf_reader\data'

# Load the PDF documents
documents = load_pdf(extracted_data)

# Split the documents into text chunks
text_chunks = text_split(documents)

#print("Length of my chunks:", len(text_chunks))


# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Save the model to the local directory
#model.save(r'D:\VS code files\building_energy_consumption\pdf_reader\model')


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings = download_hugging_face_embeddings()

#printing embeddings
#embeddings

query_result = embeddings.embed_query("Hello world")
#print("Length", len(query_result))

text_embeddings = [embeddings.embed_query(t.page_content) for t in text_chunks]
#Initializing the Pinecone
pc = Pinecone(
                api_key=PINECONE_API_KEY,
                host=PINECONE_API_HOST,
                environment="us-east-1" 
)

#index name
index_name = "aiagent"

# Connect to the index
index = pc.Index(index_name,host=PINECONE_API_HOST)

# Upsert embeddings into the index
items = [(str(i), embedding) for i, embedding in enumerate(text_embeddings)]
index.upsert(items)

#print("Embeddings have been created and stored in Pinecone.")

# Function to search the index with a query
def search_index(query):
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Ensure the embedding is in the correct format (list of floats)
    if not all(isinstance(x, float) for x in query_embedding):
        raise ValueError("Query embedding is not in the correct format. Ensure it's a list of floats.")
    
    # Search the index
    result = index.query(vector=query_embedding, top_k=1, include_values=True, include_metadata=True)
    return result

# Example query
query = "What is Etabloc?"
result = search_index(query)

# Print the results
for match in result['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}, Content: {text_chunks[int(match['id'])].page_content}")