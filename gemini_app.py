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
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    embeddings = HuggingFaceEmbeddings(model_name = os.getenv('EMBEDDING_MODEL'))
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

# Function to search the index with a query
def search_index(query, model, index):
    query_embedding = model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=10, include_values=True, include_metadata=True)
    return result

#Pinecone API
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_HOST = os.getenv('PINECONE_API_HOST')
index_name = os.getenv('INDEX_NAME')

#Initiate Pinecone.
pc = Pinecone(
                api_key=PINECONE_API_KEY,
                host=PINECONE_API_HOST,
                environment=os.getenv("ENVIRONMENT_REGION") 
)

#index name
index_name = os.getenv('INDEX_NAME')

# Connect to the index
index = pc.Index(index_name,host=PINECONE_API_HOST)

# Load the embedding model
model = SentenceTransformer(os.getenv('EMBEDDING_MODEL'))
embeddings = download_hugging_face_embeddings()

# Function to get a detailed response from Gemini
def get_gemini_response(context, question):
    prompt_template = f"""
    Please answer the following question in a detailed manner. The question is: {question}.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = prompt_template.format(context=context, question=question)
    
    model = genai.GenerativeModel('gemini-1.0-pro-latest')
    response = model.generate_content(prompt)
    
    # Check if response is valid
    if response and response.text:
        return response.text
    
    return "No response generated."

# Streamlit UI
st.title("PDF Query App")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your query")

if uploaded_files and query:
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    documents = load_pdf(temp_dir)
    text_chunks = text_split(documents)
    text_embeddings = [embeddings.embed_query(t.page_content) for t in text_chunks]
    items = [(str(i), embedding) for i, embedding in enumerate(text_embeddings)]
    index.upsert(items)

    result = search_index(query, model, index)

    if result['matches']:
        top_matches = result['matches'][:5]  # Get the top 5 matches
        st.write("Top Results:")
        for match in top_matches:
            st.write(f"ID: {match['id']}, Score: {match['score']}")
            st.write(f"Content: {text_chunks[int(match['id'])].page_content}")

        context = " ".join([text_chunks[int(match['id'])].page_content for match in top_matches])
        detailed_response = get_gemini_response(context=context, question=query)
        
        st.write(f"This is query {query}:")
        
        if detailed_response:
            st.write("Detailed Response from Gemini:")
            st.write(detailed_response)

    for uploaded_file in uploaded_files:
        os.remove(os.path.join(temp_dir, uploaded_file.name))
else:
    st.write("Please upload PDF files and enter a query.")