import os
import torch
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv

# Setting up Logging
logger = logging.getLogger("llm_loader")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger_files/model_loading.log", mode="w")  # Use 'w' mode to overwrite the log file each time
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)

# Load Environment Variables
try:
    load_dotenv()
    API_KEY = os.getenv("huggingface_token_write")
    if not API_KEY:
        logger.warning("Hugging Face API token not found in environment variables.")
except Exception as e:
    logger.error("Failed to load environment variables: %s", str(e))

# Setup the device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


# Create a checkpoint for the model to be used
checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
try:
    tokenizer_SmolLM = AutoTokenizer.from_pretrained(checkpoint)
    model_SmolLM = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    logger.info(f"Loaded SmolLM model from checkpoint: {checkpoint}")
except Exception as e:
    logger.error("Error loading SmolLM model/tokenizer: %s", str(e))


# Loading Google Gemma Model
model_id = "google/gemma-2b-it"  # Or "gemma-7b-it" if you have the adequate resources
try:
    tokenizer_Gemma = AutoTokenizer.from_pretrained(model_id, token=API_KEY)
    model_Gemma = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=API_KEY).to(device)
    logger.info(f"Loaded Gemma model: {model_id}")
except Exception as e:
    logger.error("Error loading Gemma model/tokenizer: %s", str(e))