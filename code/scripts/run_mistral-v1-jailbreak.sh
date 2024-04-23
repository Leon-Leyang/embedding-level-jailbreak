export CUDA_VISIBLE_DEVICES=0
HF_MODELS="/home/Newdisk2/jinhaibo/LLM-Safeguard/model"
model_name=Mistral-7B-Instruct-v0.1
model=${HF_MODELS}/mistralai/${model_name}

source scripts/run_meta_jailbreak.sh
