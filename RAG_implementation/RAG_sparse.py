import numpy as np
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from vector_database_creator import documents, bm25  # Import documents and BM25 model from vector_database_creator.py
from pipeline import llm  # As llm is defined in pipeline.py


# Setting up Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("RAG_sparse.log"), logging.StreamHandler()]
)

# Sparse Retrieval using loaded bm25 model
def sparse_retrieval_bm25(query, k=5):
    try:
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        retrieved_docs = [documents[i] for i in top_k_indices]
        logging.info(f"Retrieved {len(retrieved_docs)} documents using BM25 for query: '{query}'")
        return retrieved_docs
    except Exception as e:
        logging.error(f"Error during sparse retrieval: {e}")
        return []


# Create a prompt template
try:
    template = """Answer the question referring to the following context, to make your response more accurate:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    logging.info("Prompt template created successfully.")
except Exception as e:
    logging.error(f"Error creating prompt template: {e}")
    prompt = None


def format_docs(docs):
    try:
        content = "\n\n".join(doc.page_content for doc in docs)
        logging.debug("Formatted retrieved documents into context string.")
        return content
    except Exception as e:
        logging.error(f"Error formatting documents: {e}")
        return ""


# Create a sparse retrieval chain
try:
    if prompt and llm:
        sparse_qa_chain = (
            {"context": lambda x: format_docs(sparse_retrieval_bm25(x["question"])),
                "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        logging.info("Sparse QA chain created successfully.")
    else:
        sparse_qa_chain = None
        logging.warning("Prompt or LLM is missing.")
except Exception as e:
    logging.error(f"Error creating sparse QA chain: {e}")
    sparse_qa_chain = None


# User query
query = "how to establish, verify and troubleshoot vpn connection? in detail."
try:
    if sparse_qa_chain:
        response = sparse_qa_chain.invoke({"question": query})
        print(f"Question: {query}")
        print(f"Response: {response}")
        logging.info("Sparse QA chain invoked successfully.")
    else:
        logging.warning("Chain is not initialized.")
except Exception as e:
    logging.error(f"Error during chain invocation: {e}")