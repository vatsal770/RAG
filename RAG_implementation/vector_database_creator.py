from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import pickle
import os

# Load CSV file
csv_loader = CSVLoader("/home/vatsal/Documents/VS Code/RAG_demo/rag_sample_qas_from_kis.csv", encoding="utf-8")
csv_docs = csv_loader.load()

# Load Text file
text_loader = TextLoader("/home/vatsal/Documents/VS Code/RAG_demo/EEG.txt", encoding="utf-8")
text_docs = text_loader.load()

# Combine documents
all_documents = csv_docs + text_docs

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Load documents and split them into chunks
documents = text_splitter.split_documents(all_documents)
texts = [doc.page_content for doc in documents]  # Extract text for sparse embeddings

# Create embeddings for sparse retrieval using BM25
tokenized_texts = [text.split(" ") for text in texts]
bm25 = BM25Okapi(tokenized_texts)
with open("/home/vatsal/Documents/VS Code/RAG_demo/RAG_implementation/local_models/bm25_model.pkl", "wb") as f:
    pickle.dump(bm25, f)


# Create embeddings bge-small-en-v1.5
embed_model_dense = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",   # 384-dimensional embeddings
    model_kwargs={"device": "cpu"}  # or "cuda"
)
# Access and save the underlying SentenceTransformer model
embed_model_dense.client.save("local_models/bge-small-en-v1.5")


# Create vector store for dense embeddings
vectorstore_dense = Chroma.from_documents(
    documents=documents,
    embedding=embed_model_dense,
    persist_directory="chroma_db_dense"  # Specify a directory to persist the vector store
)

# Persist the vector store to disk
vectorstore_dense.persist()
