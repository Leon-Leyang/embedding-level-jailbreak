
system_prompt_type="all"

##echo """
#python estimate.py \
#    --system_prompt_type ${system_prompt_type} \
#    --config sampling --pretrained_model_path ${model}
##"""

prompt_length=20

##echo """
#python train_attack.py \
#    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
#    --config sampling --pretrained_model_path ${model}
##"""
#
#
##echo """
## harmful eval for malicious
#python generate.py \
#    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} \
#    --system_prompt_type ${system_prompt_type}\
#    --use_jailbreak_prompt \
#    --soft_prompt_path ./trained_prompts_attack
#
#python evaluate.py \
#    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
#    --model_names ${model_name}_with_jailbreak_${system_prompt_type}_${prompt_length}
##"""
#
#
##echo """
## harmful eval for advbench
#python generate.py \
#    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} \
#    --system_prompt_type ${system_prompt_type}\
#    --use_jailbreak_prompt \
#    --soft_prompt_path ./trained_prompts_attack
#
#python evaluate.py \
#    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
#    --model_names ${model_name}_with_jailbreak_${system_prompt_type}_${prompt_length}
##"""

##echo """
## harmful eval for custom
#python generate.py \
#    --use_sampling --n_samples 25 --pretrained_model_path ${model} \
#    --system_prompt_type ${system_prompt_type}\
#    --use_jailbreak_prompt \
#    --soft_prompt_path ./trained_prompts_attack
#
#python evaluate.py \
#    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
#    --model_names ${model_name}_with_jailbreak_${system_prompt_type}_${prompt_length}
#
##"""
## harmless eval for custom
#python generate.py \
#    --use_sampling --n_samples 25 --use_harmless --pretrained_model_path ${model} \
#    --system_prompt_type ${system_prompt_type}\
#    --use_jailbreak_prompt \
#    --soft_prompt_path ./trained_prompts_attack
#
#python evaluate.py \
#    --config sampling --use_harmless --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
#    --model_names ${model_name}_with_jailbreak_${system_prompt_type}_${prompt_length}

#"""
# eval for testset
python generate.py \
    --use_sampling --n_samples 25 --use_harmless --use_testset --pretrained_model_path ${model} \
    --system_prompt_type ${system_prompt_type}\
    --use_jailbreak_prompt \
    --soft_prompt_path ./trained_prompts_attack

python evaluate.py \
    --config sampling --use_harmless --use_testset --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_jailbreak_${system_prompt_type}_${prompt_length}
