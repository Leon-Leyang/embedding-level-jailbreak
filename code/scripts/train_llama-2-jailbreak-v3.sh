export CUDA_VISIBLE_DEVICES=0
HF_MODELS="/home/Newdisk2/jinhaibo/LLM-Safeguard/model"
full_model_names=(
    "meta-llama/Llama-2-7b-chat-hf"
)

for full_model_name in ${full_model_names[@]}; do
model=${HF_MODELS}/${full_model_name}
model_name=$(basename ${full_model_name})

source scripts/train_meta-jailbreak-v3.sh

done
