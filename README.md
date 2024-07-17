# PDF Query App

This application allows users to upload PDF files, extract text using OCR, and query the extracted text using a search index. The app uses various machine learning models and APIs to provide detailed responses to user queries.

*** gemini_app.py *** is the final file.


## Features

- Upload multiple PDF files
- Extract text from PDFs using OCR
- Split extracted text into chunks
- Embed text chunks using Hugging Face embeddings
- Search the text chunks using Pinecone
- Generate detailed responses using Google Generative AI

## Dependencies

To run this application, you need to install the following dependencies:

- `streamlit`
- `tensorflow`
- `langchain`
- `sentence-transformers`
- `pinecone-client`
- `python-dotenv`
- `google-generativeai`
- `doctr`
- `rapidfuzz`

You can install these dependencies using pip:



## Environment Variables

Create a `.env` file in the root directory of your project and add the following environment variables:



## Usage

1. Clone the repository:

2. Install the dependencies:

3. Create a `.env` file and add your API keys and other environment variables.

4. Run the Streamlit app:

5. Open your web browser and go to `http://localhost:8501` to use the app.

## Models Used

- **OCR Model**: `doctr` for extracting text from PDF files.
- **Embedding Model**: `Hugging Face` embeddings for text chunk embedding.
- **Search Index**: `Pinecone` for indexing and searching text chunks.
- **Generative Model**: `Google Generative AI` for generating detailed responses.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [LangChain](https://www.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Pinecone](https://www.pinecone.io/)
- [Doctr](https://mindee.github.io/doctr/)
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)
- [Google Generative AI](https://ai.google/)

## Docker Pull Command:

docker pull aakashsyadav1999/embedding_chat_bot

## Verify the Image:
docker images

## Docker Run Command:
docker run -p 8501:8501 aakashsyadav1999/embedding_chat

## Check Logs if Issues Arise:
docker ps -a  # Find your container ID
docker logs <container_id>

## Run in Interactive Mode (if needed):
docker run -it -p 8501:8501 aakashsyadav1999/embedding_chat_bot