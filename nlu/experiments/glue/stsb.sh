#!/bin/bash

cache_dir=/tmp/DeBERTa/
base_model=deberta-v3-base
task=STS-B

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $base_model \
	--do_train \
	--do_eval \
	--do_predict \
	--task_name $Task \
	--dump_interval 100 \
	--num_train_epochs 27 \
	--fp16 True \
	--warmup 50 \
	--learning_rate 5e-3 \
	--train_batch_size 32 \
	--cls_drop_out 0.1 \
	--task_name $task \
	--data_dir $cache_dir/glue_tasks/$task \
	--init_model $base_model \
	--output_dir $cache_dir/outputs/$base_model/$task
