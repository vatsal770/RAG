import logging

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from llm_loader import model_Gemma, tokenizer_Gemma

# Setting up Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline_setup.log"), logging.StreamHandler()]
)


# Forming HuggingFace Pipeline
try:
    logging.info(f"Tokenizer type: {type(tokenizer_Gemma)}")
    pipe = pipeline("text-generation", model=model_Gemma, tokenizer=tokenizer_Gemma, max_new_tokens=512)
    logging.info("Successfully created Hugging Face text-generation pipeline.")
except Exception as e:
    logging.error("Failed to initialize Hugging Face pipeline: %s", str(e))
    pipe = None 

# Forming Langchain Pipeline
try:
    if pipe is not None:
        llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("Successfully wrapped pipeline in LangChain HuggingFacePipeline.")
    else:
        llm = None
        logging.warning("Pipeline was not initialized. Skipping LangChain LLM setup.")
except Exception as e:
    logging.error("Failed to wrap pipeline in LangChain HuggingFacePipeline: %s", str(e))
    llm = None
