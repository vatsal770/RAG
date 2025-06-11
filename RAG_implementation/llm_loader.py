import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer_SmolLM = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model_SmolLM = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

###########################################################################################################################

# Load environment variables
load_dotenv()
API_KEY = os.getenv("huggingface_token_write")

model_id = "google/gemma-2b-it"  # Or "gemma-7b-it" if you have the resources
tokenizer_Gemma = AutoTokenizer.from_pretrained(model_id,token=API_KEY)

model_Gemma = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16,token = API_KEY)