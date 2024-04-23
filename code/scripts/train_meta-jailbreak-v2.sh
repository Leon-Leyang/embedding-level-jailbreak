
system_prompt_type="all"

#echo """
#python estimate.py \
#    --system_prompt_type ${system_prompt_type} \
#    --config sampling --pretrained_model_path ${model}
#"""

prompt_length=20

#echo """
python train_attack.py \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --config sampling --pretrained_model_path ${model}
#"""


#echo """
# harmful eval for malicious
python generate.py \
    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} \
    --system_prompt_type ${system_prompt_type}\
    --use_jailbreak_prompt \
    --soft_prompt_path ./trained_prompts_attack-v2

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_jailbreak-v2_${system_prompt_type}_${prompt_length}
#"""


#echo """
# harmful eval for advbench
python generate.py \
    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} \
    --system_prompt_type ${system_prompt_type}\
    --use_jailbreak_prompt \
    --soft_prompt_path ./trained_prompts_attack-v2

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_jailbreak-v2_${system_prompt_type}_${prompt_length}
#"""
