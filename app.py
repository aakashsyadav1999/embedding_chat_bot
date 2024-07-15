import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification,AutoModelForSeq2SeqLM
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

# Function to load LLaMA 2 model and tokenizer with error handling
def load_llama_model():
    try:
        token = "YOUR_HUGGING_FACE_TOKEN"
        llama_tokenizer = AutoTokenizer.from_pretrained("describeai/gemini", token=token)
        llama_model = AutoModelForSeq2SeqLM.from_pretrained("describeai/gemini", token=token)
        return llama_tokenizer, llama_model
    except Exception as e:
        st.error(f"Error loading distilbert/distilgpt2: {e}")
        return None, None

llama_tokenizer, llama_model = load_llama_model()

# Function to generate response using LLaMA 2
def generate_llama_response(prompt, max_length=1024, max_new_tokens=300, temperature=0.1, top_p=0.9, repetition_penalty=1.2):
    if llama_tokenizer and llama_model:
        try:
            # Tokenize the prompt and ensure it's within the maximum length
            inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
            
            # Generate text
            outputs = llama_model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=llama_tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
            )
            
            # Decode the generated text
            response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove simple repetitions
            response = ' '.join(dict.fromkeys(response.split()))
            
            return response
        except Exception as e:
            st.error(f"Error generating response from LLaMA 2: {e}")
            return None
    else:
        st.error("LLaMA 2 model is not loaded.")
        return None

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

    result = search_index(query, model, index, text_chunks)

    if result['matches']:
        top_match = result['matches'][0]
        st.write("Top Result:")
        st.write(f"ID: {top_match['id']}, Score: {top_match['score']}")
        st.write(f"Content: {text_chunks[int(top_match['id'])].page_content}")

        llama_response = generate_llama_response(text_chunks[int(top_match['id'])].page_content)
        if llama_response:
            st.write("Detailed Response from LLaMA 2:")
            st.write(llama_response)

    for uploaded_file in uploaded_files:
        os.remove(os.path.join(temp_dir, uploaded_file.name))
else:
    st.write("Please upload PDF files and enter a query.")