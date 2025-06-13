from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from llm_loader import model_Gemma, tokenizer_Gemma


print(type(tokenizer_Gemma))
pipe = pipeline("text-generation", model=model_Gemma, tokenizer=tokenizer_Gemma, max_new_tokens=512)

llm = HuggingFacePipeline(pipeline=pipe)