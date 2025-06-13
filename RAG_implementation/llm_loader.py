import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv


# Create a checkpoint for the model to be used
checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"

# Setup the device
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_SmolLM = AutoTokenizer.from_pretrained(checkpoint)
# in-case of GPU support - install accelerate , and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model_SmolLM = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)



# Load environment variables
load_dotenv()
API_KEY = os.getenv("huggingface_token_write")

model_id = "google/gemma-2b-it"  # Or "gemma-7b-it" if you have the adequate resources
tokenizer_Gemma = AutoTokenizer.from_pretrained(model_id,token=API_KEY)

model_Gemma = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16,token = API_KEY).to(device)