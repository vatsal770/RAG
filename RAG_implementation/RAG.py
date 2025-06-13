'''RAG Implementation - LLM Configuration'''

from langchain.chains import RetrievalQA

from pipeline import llm  # As llm is defined in pipeline.py
from vector_database_creator import vectorstore_dense  # Importing the embedding model from vector_database_creator.py



# Create retriever
retriever = vectorstore_dense.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Create RetrievalQA instance
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # stuff, map_reduce, refine, or map_rerank are the available chain types
    retriever=retriever,
    return_source_documents=True  # Set to True if you want to return source documents
)


# Function to clean the response from the LLM
def clean_response(raw_response, marker="Helpful Answer:"):
    if marker in raw_response:
        return raw_response.split(marker)[-1].strip()
    return raw_response.strip()


# Test the RetrievalQA instance with a query
query = "What is the purpose of an EEG in detail and what can you expect to happen during the procedure?"
# query = "how to establish, verify and troubleshoot vpn connection? in detail."

response = retrieval_qa.invoke(query)
cleaned = clean_response(response['result'])
print(f"Question: {query}")
print(f"Response: {cleaned}")

print("\n" + "#"*50 + "\n")
print("Source Documents:")
for doc in response['source_documents']:
    print(doc.page_content)


