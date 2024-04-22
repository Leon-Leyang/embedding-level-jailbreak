export CUDA_VISIBLE_DEVICES=0,3,5
HF_MODELS="/home/Newdisk2/jinhaibo/LLM-Safeguard/model"
model_name=Llama-2-7b-chat-hf
model=${HF_MODELS}/meta-llama/${model_name}

source scripts/run_meta_jailbreak.sh
