import os
from dotenv import load_dotenv

load_dotenv('.env')
token = os.getenv('token')

from transformers import pipeline

pipe = pipeline('text-generation', "c2p-cmd/gemma-2-2b-quote-generator", token=token, device_map="mps")

print(pipe("Quote:", max_new_tokens=40))