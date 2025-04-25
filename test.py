from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from duckduckgo_search import DDGS  # or use SerpAPI, Google, etc.
import torch

# Load a small model (for speed) - swap for bigger ones locally
model_name = "tiiuae/falcon-7b-instruct"
# change to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct when given access

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Wrap in a text generation pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to search the web
def web_search(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["body"])
    return "\n".join(results)

# Main loop
def ask_llm_with_search(prompt):
    search_results = web_search(prompt)
    full_prompt = f"Use the following information from the web to answer:\n{search_results}\n\nQuestion: {prompt}\nAnswer:"
    result = llm(full_prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
    return result

# Example usage
query = "What are the latest advancements in Alzheimer's treatment?"
response = ask_llm_with_search(query)
print(response)
