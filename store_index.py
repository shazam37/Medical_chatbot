from langchain.pinecone import PineconeVectorStore
from src.helper import load_pdf, text_splitter, download_hugging_face_embeddings

from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initializing the pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

text_field = [text.page_content for text in text_chunks]
index_name = "medical-chatbot"

vectorstore = PineconeVectorStore.from_texts(
    text_field, embeddings, index_name=index_name
)

