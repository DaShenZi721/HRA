BASE_MODEL="/root/autodl-tmp/Finetune/Model/7B/"
OUTPUT="output/checkpoint"

python finetune_32.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --hrft_r 32 \
    --init_a 1e-4 \
    --eps 1e-4 \
    --add_orth "none" \
    --data_path "/root/autodl-tmp/Finetune/Data/MATHQA/" \
    --dataset_split "train[:100000]"\
    --dataset_field query response \
    --num_train_epochs 2 \
    --per_device_train_batch_size 30 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.005 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to "wandb"