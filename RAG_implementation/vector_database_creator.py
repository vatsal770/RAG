import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# Load CSV file
file_path_csv = os.path.join(os.getcwd(), "documents", "Setting_Up_Mobile_Device_for_Company_Email.csv")
csv_loader = CSVLoader(file_path_csv, encoding="utf-8")
csv_docs = csv_loader.load()


# Load Text file
file_path_text = os.path.join(os.getcwd(), "documents", "EEG.txt")
text_loader = TextLoader(file_path_text, encoding="utf-8")
text_docs = text_loader.load()


# Combine documents
all_documents = csv_docs + text_docs


# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])


# Load documents and split them into chunks
documents = text_splitter.split_documents(all_documents)
texts = [doc.page_content for doc in documents]  # Extract text for sparse embeddings


# Create embeddings for sparse retrieval using BM25 Model
tokenized_texts = [text.split(" ") for text in texts]
bm25 = BM25Okapi(tokenized_texts)


# Create embeddings 
embed_model_dense = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",   # 384-dimensional embeddings
    model_kwargs={"device": "cpu"}  # or "cuda"
)


# Create vector store for dense embeddings
vectorstore_dense = Chroma.from_documents(
    documents=documents,
    embedding=embed_model_dense,
)
