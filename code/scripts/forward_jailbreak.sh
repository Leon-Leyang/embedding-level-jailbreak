export CUDA_VISIBLE_DEVICES=0,1
HF_MODELS="/home/Newdisk2/jinhaibo/LLM-Safeguard/model"
full_model_name="meta-llama/Llama-2-7b-chat-hf"


model=${HF_MODELS}/${full_model_name}

# Forward train set + attack
python forward_with_jailbreak.py \
    --pretrained_model_path ${model} \
    --jailbreak_prompt_path ./trained_prompts_attack \
    --output_path ./hidden_states_with_jailbreak

# Forward test set
python forward.py \
    --use_testset \
    --pretrained_model_path ${model}

# Forward test set + attack
python forward_with_jailbreak.py \
    --use_testset \
    --pretrained_model_path ${model} \
    --jailbreak_prompt_path ./trained_prompts_attack \
    --output_path ./hidden_states_with_jailbreak

# Forward advbench
python forward.py \
    --use_advbench \
    --pretrained_model_path ${model}

# Forward advbench + attack
python forward_with_jailbreak.py \
    --use_advbench \
    --pretrained_model_path ${model} \
    --jailbreak_prompt_path ./trained_prompts_attack \
    --output_path ./hidden_states_with_jailbreak

# Forward malicious
python forward.py \
    --use_malicious \
    --pretrained_model_path ${model}

# Forward malicious + attack
python forward_with_jailbreak.py \
    --use_malicious \
    --pretrained_model_path ${model} \
    --jailbreak_prompt_path ./trained_prompts_attack \
    --output_path ./hidden_states_with_jailbreak
#"""

