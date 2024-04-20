export OPENAI_API_KEY=$(< ./api_key)

alpaca_eval evaluate --model_outputs './outputs_alpaca/Mistral-7B-Instruct-v0.1_with_soft_all_default_alpaca/output_sampling.csv' \
                     -r './data_alpaca/alpaca_eval.json' \
                     --annotators_config 'alpaca_eval_gpt4'

