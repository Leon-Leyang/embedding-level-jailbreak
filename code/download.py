from transformers import AutoModelForCausalLM, AutoTokenizer

# List of models to initialize and save
model_names = [
    "meta-llama/Llama-2-7b-chat-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "lmsys/vicuna-7b-v1.5",
    "microsoft/Orca-2-7b",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat/openchat-3.5",
    "openchat/openchat-3.5-1210",
    "meta-llama/LlamaGuard-7b"
]


def save_model_and_tokenizer(model_name, cache_dir):
    # Initialize model and tokenizer from the cache
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


# Iterate over each model and save them to the final directory
cache_dir = "D:/models"
for model_name in model_names:
    print(f"Initializing and saving {model_name}...")
    save_model_and_tokenizer(model_name, cache_dir)

print("All models have been initialized and saved locally.")

