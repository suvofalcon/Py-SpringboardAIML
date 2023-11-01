import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
import pinecone
from langchain.vectorstores import Pinecone


#**********Functions to help you load documents to PINECONE***********

# Read PDF Data
def read_pdf_data(pdf_file):
    pdf_doc = PdfReader(pdf_file)
    text=""
    for page in pdf_doc.pages:
        text += page.extract_text()

    return text

# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text=text)
    docs_chunks = text_splitter.create_documents(docs)
    return docs_chunks

# Create embeddings instance
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# Function to push data to pinecone
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):

    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )

    index = Pinecone.from_documents(docs, embedding=embeddings, index_name=pinecone_index_name)
    return index



