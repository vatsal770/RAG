import numpy as np
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever

from vector_database_creator import documents  # Import BM25 model from vector_database_creator.py
from pipeline import llm  # As llm is defined in pipeline.py



# Setting up Logging
logger = logging.getLogger("rag_sparse")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger_files/RAG_sparse.log", mode="w")  # Use 'w' mode to overwrite the log file each time
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


# Create BM25 retriever
try:
    bm25_retriever = BM25Retriever.from_documents(documents=documents, k=5)
    logger.info("BM25 retriever initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing BM25 retriever: {e}")
    bm25_retriever = None

# # Sparse Retrieval using loaded bm25 model
# def sparse_retrieval_bm25(query, k=5):
#     try:
#         tokenized_query = query.split(" ")
#         doc_scores = bm25.get_scores(tokenized_query)
#         top_k_indices = np.argsort(doc_scores)[-k:][::-1]
#         retrieved_docs = [documents[i] for i in top_k_indices]
#         logger.info(f"Retrieved {len(retrieved_docs)} documents using BM25 for query: '{query}'")
#         return retrieved_docs
#     except Exception as e:
#         logger.error(f"Error during sparse retrieval: {e}")
#         return []


# Create a prompt template
try:
    template = """Answer the question referring to the following context, to make your response more accurate:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created successfully.")
except Exception as e:
    logger.error(f"Error creating prompt template: {e}")
    prompt = None


def format_docs(docs):
    try:
        content = "\n\n".join(doc.page_content for doc in docs)
        logger.debug("Formatted retrieved documents into context string.")
        return content
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        return ""


# Create a sparse retrieval chain
try:
    if bm25_retriever and prompt and llm:
        sparse_qa_chain = (
            {"context": lambda x: format_docs(bm25_retriever.get_relevant_documents(x["question"])),
                "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        logger.info("Sparse QA chain created successfully.")
    else:
        sparse_qa_chain = None
        logger.warning("Missing components: retriever, prompt, or llm.")
except Exception as e:
    logger.error(f"Error creating sparse QA chain: {e}")
    sparse_qa_chain = None


# User query
if __name__ == "__main__":
    query = "how to establish, verify and troubleshoot vpn connection? in detail."
    try:
        if sparse_qa_chain:
            response = sparse_qa_chain.invoke({"question": query})
            print(f"Question: {query}")
            print(f"Response: {response}")
            logger.info("Sparse QA chain invoked successfully.")
        else:
            logger.warning("Chain is not initialized.")
    except Exception as e:
        logger.error(f"Error during chain invocation: {e}")