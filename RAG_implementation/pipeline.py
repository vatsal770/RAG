import logging

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from llm_loader import model_Gemma, tokenizer_Gemma

# Setting up Logging
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger_files/pipeline_setup.log", mode="w")  # Use 'w' mode to overwrite the log file each time
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


# Forming HuggingFace Pipeline
try:
    logger.info(f"Tokenizer type: {type(tokenizer_Gemma)}")
    pipe = pipeline("text-generation", model=model_Gemma, tokenizer=tokenizer_Gemma, max_new_tokens=512)
    logger.info("Successfully created Hugging Face text-generation pipeline.")
except Exception as e:
    logger.error("Failed to initialize Hugging Face pipeline: %s", str(e))
    pipe = None 

# Forming Langchain Pipeline
try:
    if pipe is not None:
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Successfully wrapped pipeline in LangChain HuggingFacePipeline.")
    else:
        llm = None
        logger.warning("Pipeline was not initialized. Skipping LangChain LLM setup.")
except Exception as e:
    logger.error("Failed to wrap pipeline in LangChain HuggingFacePipeline: %s", str(e))
    llm = None
