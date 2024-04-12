import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the cache directory
os.environ["TRANSFORMERS_CACHE"] = "/home/Newdisk2/jinhaibo/cache"

# Define where the models should be finally saved
final_model_directory = "/home/Newdisk2/jinhaibo/LLM-Safeguard/model"

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


def save_model_and_tokenizer(model_name, base_directory):
    save_directory = os.path.join(base_directory, model_name.split('/')[0], model_name.split('/')[1])
    # Ensure save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Initialize model and tokenizer from the cache
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save model and tokenizer to the final directory
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Saved {model_name} to {save_directory}")


# Iterate over each model and save them to the final directory
for model_name in model_names:
    print(f"Initializing and saving {model_name}...")
    save_model_and_tokenizer(model_name, final_model_directory)

print("All models have been initialized and saved locally.")

