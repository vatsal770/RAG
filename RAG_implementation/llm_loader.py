import os
import torch
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv

# Setting up Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("model_loading.log"), logging.StreamHandler()]
)

# Load Environment Variables
try:
    load_dotenv()
    API_KEY = os.getenv("huggingface_token_write")
    if not API_KEY:
        logging.warning("Hugging Face API token not found in environment variables.")
except Exception as e:
    logging.error("Failed to load environment variables: %s", str(e))

# Setup the device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")


# Create a checkpoint for the model to be used
checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
try:
    tokenizer_SmolLM = AutoTokenizer.from_pretrained(checkpoint)
    model_SmolLM = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    logging.info(f"Loaded SmolLM model from checkpoint: {checkpoint}")
except Exception as e:
    logging.error("Error loading SmolLM model/tokenizer: %s", str(e))


# Loading Google Gemma Model
model_id = "google/gemma-2b-it"  # Or "gemma-7b-it" if you have the adequate resources
try:
    tokenizer_Gemma = AutoTokenizer.from_pretrained(model_id, token=API_KEY)
    model_Gemma = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=API_KEY).to(device)
    logging.info(f"Loaded Gemma model: {model_id}")
except Exception as e:
    logging.error("Error loading Gemma model/tokenizer: %s", str(e))