import logging

from langchain_core.prompts import PromptTemplate

from pipeline import llm  # As llm is defined in pipeline.py


# Setting up Logging
logger = logging.getLogger("llm_response")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger_files/model_response.log", mode="w")  # Use 'w' mode to overwrite the log file each time
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)

# Create a prompt template
try:
    template = """You are a helpful assistant. Give answer to the question to your best knowledge.

Question: {question}

Answer: Let's think step by step."""
    
    prompt = PromptTemplate.from_template(template)
    logger.info("PromptTemplate successfully created.")
except Exception as e:
    logger.error("Error creating PromptTemplate: %s", str(e))
    prompt = None

# Building a chain
try:
    if prompt is not None and llm is not None:
        chain = prompt | llm
        logger.info("Successfully created prompt → LLM chain.")
    else:
        chain = None
        logger.warning("Chain creation skipped due to missing prompt or LLM.")
except Exception as e:
    logger.error("Error building the chain: %s", str(e))
    chain = None


# User prompt Question
question = "how to establish, verify and troubleshoot vpn connection? in detail."

try:
    if chain is not None:
        result = chain.invoke({"question": question})
        print(result)
        logger.info("Chain invoked successfully.")
    else:
        logger.warning("Chain is not initialized. Cannot run inference.")
except Exception as e:
    logger.error("Error during chain invocation: %s", str(e))
