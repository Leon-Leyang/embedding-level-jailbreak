
system_prompt_type="all"

#echo """
python estimate.py \
    --system_prompt_type ${system_prompt_type} \
    --config sampling --pretrained_model_path ${model}
#"""

prompt_length="default"

#echo """
python train_ortho.py \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --config sampling --pretrained_model_path ${model}
#"""


#echo """
# testset, only for harmless
python generate.py \
    --use_sampling --n_samples 25 --use_soft_prompt --use_harmless --use_testset \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} --pretrained_model_path ${model} \
    --output_path ./outputs_ortho \
    --soft_prompt_path ./trained_prompts_ortho

python evaluate.py \
    --config sampling --use_harmless --use_testset \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length} \
    --generation_output_path ./outputs_ortho \
    --output_path ./eval_results_ortho
#"""


#echo """
# alpaca eval
python generate.py \
    --use_sampling --n_samples 1 --use_soft_prompt --use_alpaca --pretrained_model_path ${model} \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --output_path ./outputs_ortho \
    --soft_prompt_path ./trained_prompts_ortho
#"""


#echo """
# harmful eval for malicious
python generate.py \
    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} \
    --use_soft_prompt --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --output_path ./outputs_ortho \
    --soft_prompt_path ./trained_prompts_ortho

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length} \
    --generation_output_path ./outputs_ortho \
    --output_path ./eval_results_ortho
#"""


#echo """
# harmful eval for advbench
python generate.py \
    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} \
    --use_soft_prompt --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --output_path ./outputs_ortho \
    --soft_prompt_path ./trained_prompts_ortho

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length} \
    --generation_output_path ./outputs_ortho \
    --output_path ./eval_results_ortho
#"""
