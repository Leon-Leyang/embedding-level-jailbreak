export CUDA_VISIBLE_DEVICES=0
HF_MODELS="/home/Newdisk2/jinhaibo/LLM-Safeguard/model"
full_model_name=${HF_MODELS}/"meta-llama/Llama-2-7b-chat-hf"


#python compare_train_harmfulness_jailbreak.py \
#    --pretrained_model_paths ${full_model_name} \
#    --config sampling \
#    --output_path comparisons/train_jailbreak
#
#
#python compare_train_refusal_jailbreak.py \
#    --pretrained_model_paths ${full_model_name} \
#    --config sampling \
#    --output_path comparisons/train_jailbreak


python compare_ood_harmfulness_jailbreak.py \
    --pretrained_model_paths ${full_model_name} \
    --config sampling \
    --output_path comparisons/train_jailbreak


python compare_ood_refusal_jailbreak.py \
    --pretrained_model_paths ${full_model_name} \
    --config sampling \
    --output_path comparisons/train_jailbreak