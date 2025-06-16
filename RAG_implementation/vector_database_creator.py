import os
import logging

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# Setting up Logging
logger = logging.getLogger("vector_db")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger_files/vector_database.log", mode="w")  # Use 'w' mode to overwrite the log file each time
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


# Load CSV file
try:
    file_path_csv = os.path.join(os.getcwd(), "documents", "Setting_Up_Mobile_Device_for_Company_Email.csv")
    csv_loader = CSVLoader(file_path_csv, encoding="utf-8")
    csv_docs = csv_loader.load()
    logger.info("Loaded CSV file successfully.")
except Exception as e:
    logger.error("Error loading CSV file: %s", str(e))
    csv_docs = []


# Load Text file
try:
    file_path_text = os.path.join(os.getcwd(), "documents", "EEG.txt")
    text_loader = TextLoader(file_path_text, encoding="utf-8")
    text_docs = text_loader.load()
    logger.info("Loaded text file successfully.")
except Exception as e:
    logger.error("Error loading text file: %s", str(e))
    text_docs = []


# Combine documents
all_documents = csv_docs + text_docs
logger.info(f"Total documents loaded: {len(all_documents)}")

# Split documents into chunks
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    documents = text_splitter.split_documents(all_documents)
    logger.info(f"Documents successfully split into {len(documents)} chunks.")
except Exception as e:
    logger.error("Error during document splitting: %s", str(e))
    documents = []


# Create embeddings for sparse retrieval using BM25 Model
try:
    texts = [doc.page_content for doc in documents]
    tokenized_texts = [text.split(" ") for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    logger.info("BM25 sparse retriever initialized.")
except Exception as e:
    logger.error("Error creating BM25 retriever: %s", str(e))
    bm25 = None


# Create embeddings 
try:
    embed_model_dense = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # 384-dimensional embeddings
        model_kwargs={"device": "cpu"}        # or "cuda"
    )
    logger.info("Dense embedding model loaded successfully.")
except Exception as e:
    logger.error("Error loading dense embedding model: %s", str(e))
    embed_model_dense = None


# Create vector store for dense embeddings
try:
    if embed_model_dense and documents:
        vectorstore_dense = Chroma.from_documents(documents=documents, embedding=embed_model_dense)
        logger.info("Dense vector store created successfully.")
    else:
        vectorstore_dense = None
        logger.warning("Skipped vector store creation due to missing documents or embedding model.")
except Exception as e:
    logger.error("Error creating dense vector store: %s", str(e))
    vectorstore_dense = None
