BASE_MODEL="/root/autodl-tmp/Finetune/Model/7B/"
OUTPUT="output/checkpoint"

python merge_adapter_to_base_model.py --base_mode $BASE_MODEL --adapter $OUTPUT/ft/ --output_path $OUTPUT
python inference/gsm8k_inference.py --model $OUTPUT
python inference/MATH_inference.py --model $OUTPUT