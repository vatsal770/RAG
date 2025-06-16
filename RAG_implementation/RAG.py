'''RAG Implementation - LLM Configuration'''

import logging

from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever

from pipeline import llm  # As llm is defined in pipeline.py
from vector_database_creator import vectorstore_dense  # Importing the embedding model from vector_database_creator.py
from RAG_sparse import bm25_retriever  # Importing the BM25 retriever from RAG_sparse.py


# Setting up logger
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger_files/RAG.log", mode="w")  # Use 'w' mode to overwrite the log file each time
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


# Create retriever
try:
    retriever = vectorstore_dense.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # Initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
                                       weights=[0.4, 0.6])
    logger.info("Retriever created successfully from dense vectorstore.")
except Exception as e:
    logger.error(f"Error creating retriever: {e}")
    retriever = None

# Create RetrievalQA instance
try:
    if llm and retriever:
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # stuff, map_reduce, refine, or map_rerank are the available chain types
            retriever=retriever,
            return_source_documents=True  # Set to True if you want to return source documents
        )
        logger.info("RetrievalQA chain created successfully.")
    else:
        retrieval_qa = None
        logger.warning("RetrievalQA chain creation skipped due to missing LLM or retriever.")
except Exception as e:
    logger.error(f"Error creating RetrievalQA: {e}")
    retrieval_qa = None


# Function to clean the response from the LLM
def clean_response(raw_response, marker="Helpful Answer:"):
    if marker in raw_response:
        return raw_response.split(marker)[-1].strip()
    return raw_response.strip()


# Test the RetrievalQA instance with a query
if __name__ == "__main__":
    logger.info("Starting RetrievalQA chain usage for user query...")
    query = "What is the purpose of an EEG in detail and what can you expect to happen during the procedure?"
    # query = "how to establish, verify and troubleshoot vpn connection? in detail."

    try:
        if retrieval_qa:
            response = retrieval_qa.invoke(query)
            cleaned = clean_response(response['result'])

            print(f"Question: {query}")
            print(f"Response: {cleaned}")

            logger.info("RAG chain invoked and response printed.")

            print("\n" + "#" * 50 + "\n")
            print("Source Documents:")
            for doc in response['source_documents']:
                print(doc.page_content)
        else:
            logger.warning("retrieval_qa is not initialized. Skipping invocation.")
    except Exception as e:
        logger.error(f"Error invoking RetrievalQA chain: {e}")



