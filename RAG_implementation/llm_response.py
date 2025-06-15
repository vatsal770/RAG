import logging

from langchain_core.prompts import PromptTemplate

from pipeline import llm  # As llm is defined in pipeline.py


# Setting up Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("llm_response.log"), logging.StreamHandler()]
)

# Create a prompt template
try:
    template = """You are a helpful assistant. Give answer to the question to your best knowledge.

Question: {question}

Answer: Let's think step by step."""
    
    prompt = PromptTemplate.from_template(template)
    logging.info("PromptTemplate successfully created.")
except Exception as e:
    logging.error("Error creating PromptTemplate: %s", str(e))
    prompt = None

# Building a chain
try:
    if prompt is not None and llm is not None:
        chain = prompt | llm
        logging.info("Successfully created prompt → LLM chain.")
    else:
        chain = None
        logging.warning("Chain creation skipped due to missing prompt or LLM.")
except Exception as e:
    logging.error("Error building the chain: %s", str(e))
    chain = None


# User prompt Question
question = "how to establish, verify and troubleshoot vpn connection? in detail."

try:
    if chain is not None:
        result = chain.invoke({"question": question})
        print(result)
        logging.info("Chain invoked successfully.")
    else:
        logging.warning("Chain is not initialized. Cannot run inference.")
except Exception as e:
    logging.error("Error during chain invocation: %s", str(e))
