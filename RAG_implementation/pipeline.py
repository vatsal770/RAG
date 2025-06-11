from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline

# model = AutoModelForCausalLM.from_pretrained("local_models/SmolLM-1.7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("local_models/SmolLM-1.7B-Instruct")

model = AutoModelForCausalLM.from_pretrained("local_models/gemma-2b-it")
tokenizer = AutoTokenizer.from_pretrained("local_models/gemma-2b-it")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

llm = HuggingFacePipeline(pipeline=pipe)