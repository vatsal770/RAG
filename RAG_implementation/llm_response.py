from langchain_core.prompts import PromptTemplate
from pipeline import llm  # As llm is defined in pipeline.py

template = """You are a helpful assistant. Give answer to the question to your best knowledge.

Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "how to establish, verify and troubleshoot vpn connection? in detail."

print(chain.invoke({"question": question}))
