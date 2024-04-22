
#echo """
# base
python generate.py \
    --use_sampling --n_samples 25 \
    --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}
#"""

#echo """
# jailbreak with sure prompt
python generate.py \
    --use_sampling --n_samples 25 \
    --pretrained_model_path ${model} \
    --use_jailbreak

python evaluate.py \
    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name} \
    --use_jailbreak
#"""
