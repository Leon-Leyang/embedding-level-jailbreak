export CUDA_VISIBLE_DEVICES=6,7
HF_MODELS="/home/Newdisk2/jinhaibo/LLM-Safeguard/model"
full_model_names=(
    "mistralai/Mistral-7B-Instruct-v0.1"
)

for full_model_name in ${full_model_names[@]}; do
model=${HF_MODELS}/${full_model_name}
model_name=$(basename ${full_model_name})

source scripts/train_meta-reproc.sh

done
