#!/bin/bash

cache_dir=/tmp/DeBERTa/
base_model=deberta-v3-base
task=QQP

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $base_model \
	--do_train \
	--do_eval \
	--do_predict \
	--max_seq_len 320 \
	--dump_interval 500 \
	--num_train_epochs 10 \
	--fp16 True \
	--warmup 1000 \
	--learning_rate 9e-4 \
	--train_batch_size 32 \
	--cls_drop_out 0.1 \
	--task_name $task \
	--data_dir $cache_dir/glue_tasks/$task \
	--init_model $base_model \
	--output_dir $cache_dir/outputs/$base_model/$task

