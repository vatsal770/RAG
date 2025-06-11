import pickle
import numpy as np
from vector_database_creator import documents  # Importing documents from vector_database_creator.py
from pipeline import llm  # As llm is defined in pipeline.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

with open("local_models/bm25_model.pkl", "rb") as f:
    bm25 = pickle.load(f)

def sparse_retrieval_bm25(query, k=5):
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(doc_scores)[-k:][::-1]
    return [documents[i] for i in top_k_indices]

# Create a prompt template
template = """Answer the question refering to the following context, to make your response more accurate:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create a sparse retrieval chain
sparse_qa_chain = (
    {"context": lambda x: format_docs(sparse_retrieval_bm25(x["question"])), 
     "question": RunnablePassthrough()}
    | prompt
    | llm
)


query = "how to establish, verify and troubleshoot vpn connection? in detail."
response = sparse_qa_chain.invoke({"question": query})
print(f"Question: {query}")
print(f"Response: {response}")