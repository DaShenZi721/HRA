# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/home/shen_yuan/huggingface_aliendao/aliendao/dataroot/models/runwayml/stable-diffusion-v1-5"
export HF_HOME='/tmp'

# Define the unique_token, class_tokens, and subject_names
unique_token="qwe"
subject_names=(
    "backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can"
    "candle" "cat" "cat2" "clock" "colorful_sneaker"
    "dog" "dog2" "dog3" "dog5" "dog6"
    "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie"
    "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon"
    "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie"
)

class_tokens=(
    "backpack" "backpack" "stuffed animal" "bowl" "can"
    "candle" "cat" "cat" "clock" "sneaker"
    "dog" "dog" "dog" "dog" "dog"
    "dog" "dog" "toy" "boot" "stuffed animal"
    "toy" "glasses" "toy" "toy" "cartoon"
    "toy" "sneaker" "teapot" "vase" "stuffed animal"
)
# sequence_0=(
#   0 6 8 13 15 16 17 19 20 22
# )
# sequence_1=(
#   1 2 3 4 6 8 12 14 20
# )
# sequence_2=(
#   2 5 7 10 11 13 16 22 24
# )
# sequence_3=(
#   0 2 8 12 13 16 20 21
# )
# sequence_4=(
#   3 4 8 9 12 13 14 16 20 24
# )
# sequence_5=(
#   0 2 3 4 7 8 10 18 22
# )
# sequence_6=(
#   0 5 6 7 8 11 15 18 19 22
# )
# sequence_7=(
#   3 4 5 10 14 17 19 21 24
# )
# sequence_8=(
#   3 5 6 8 11 13 15 20 23
# )
# sequence_9=(
#   0 1 2 6 11 12 13 16 21
# )
# sequence_10=(
#   3 4 10 12 14 18 19 22 24
# )
# sequence_11=(
#   0 2 4 5 13 15 20 21 23
# )
# sequence_12=(
#   2 4 7 8 11 12 16 17 24
# )
# sequence_13=(
#   5 6 9 12 15 16 20 22 24
# )
# sequence_14=(
#   0 9 10 11 12 13 19 22 23
# )
# sequence_15=(
#   1 3 4 7 9 11 13 16 20
# )
# sequence_16=(
#   5 7 9 11 16 17 19 21 22
# )
# sequence_17=(
#   0 1 4 7 8 9 18 21 22
# )
# sequence_18=(
#   0 2 4 7 9 14 19 22 24
# )
# sequence_19=(
#   4 7 10 13 17 19 22 23 24
# )
# sequence_20=(
#   3 4 6 12 17 20 21 22 23
# )
# sequence_21=(
#   0 4 7 9 10 11 22 23 24
# )
# sequence_22=(
#   3 5 7 9 14 16 18 22 23
# )
# sequence_23=(
#   3 4 5 9 14 16 19 20 21
# )
# sequence_24=(
#   2 4 6 7 13 14 18 19 20
# )
# sequence_25=(
#   3 5 9 10 14 18 20 21 23
# )
# sequence_26=(
#   0 2 3 6 10 11 14 18 19
# )
# sequence_27=(
#   3 9 10 12 13 17 19 20 22
# )
# sequence_28=(
#   0 1 6 10 11 13 16 18 23
# )
# sequence_29=(
#   1 4 5 8 18 22 23 24
# )
# "backpack" 0 6 8 13 15 16 17 19 20 22
# "backpack_dog" 1 2 3 4 6 8 12 14 20
# "bear_plushie" 2 5 7 10 11 13 16 22 24
# "berry_bowl" 0 2 8 12 13 16 20 21
# "can" 3 4 8 9 12 13 14 16 20 24
# "candle" 0 2 3 4 7 8 10 18 22
# "cat" 0 5 6 7 8 11 15 18 19 22
# "cat2" 3 4 5 10 14 17 19 21 24
# "clock" 3 5 6 8 11 13 15 20 23
# "colorful_sneaker" 0 1 2 6 11 12 13 16 21
# "dog"  3 4 10 12 14 18 19 22 24
# "dog2" 0 2 4 5 13 15 20 21 23
# "dog3" 2 4 7 8 11 12 16 17 24
# "dog5" 5 6 9 12 15 16 20 22 24
# "dog6" 0 9 10 11 12 13 19 22 23
# "dog7" 1 3 4 7 9 11 13 16 20
# "dog8" 5 7 9 11 16 17 19 21 22
# "duck_toy" 0 1 4 7 8 9 18 21 22
# "fancy_boot" 0 2 4 7 9 14 19 22 24
# "grey_sloth_plushie" 4 7 10 13 17 19 22 23 24
# "monster_toy" 3 4 6 12 17 20 21 22 23
# "pink_sunglasses"  0 4 7 9 10 11 22 23 24
# "poop_emoji"  3 5 7 9 14 16 18 22 23
# "rc_car" 3 4 5 9 14 16 19 20 21
# "red_cartoon" 2 4 6 7 13 14 18 19 20
# "robot_toy" 3 5 9 10 14 18 20 21 23
# "shiny_sneaker"  0 2 3 6 10 11 14 18 19
# "teapot" 3 9 10 12 13 17 19 20 22
# "vase" 0 1 6 10 11 13 16 18 23
# "wolf_plushie" 1 4 5 8 18 22 23 24
# 生成30个序列，每个序列包含0到24的数字
for ((i=0; i<30; i++)); do
    sequence=$(seq 0 24)
    
    # 使用shuf命令对序列进行随机打乱
    shuffled_sequence=$(shuf -e $sequence)
    
    # 将随机打乱后的序列存储到变量中
    eval "sequence_$i=($shuffled_sequence)"
done

for ((j=0; j<25; j++)); do
for ((i=0; i<30; i++)); do
    
eval "current_sequence=(\${sequence_$i[@]})"

# # prompt_idx=$((idx % 25))
# # class_idx=$((idx / 25))
prompt_idx=${current_sequence[$j]}
class_idx=$i
eps=7e-6
lr=7e-6
l=7

echo "prompt_idx: $prompt_idx, class_idx: $class_idx"

class_token=${class_tokens[$class_idx]}
selected_subject=${subject_names[$class_idx]}

if [[ $class_idx =~ ^(0|1|2|3|4|5|8|9|17|18|19|20|21|22|23|24|25|26|27|28|29)$ ]]; then
  prompt_list=(
    "a ${unique_token} ${class_token} in the jungle"
    "a ${unique_token} ${class_token} in the snow"
    "a ${unique_token} ${class_token} on the beach"
    "a ${unique_token} ${class_token} on a cobblestone street"
    "a ${unique_token} ${class_token} on top of pink fabric"
    "a ${unique_token} ${class_token} on top of a wooden floor"
    "a ${unique_token} ${class_token} with a city in the background"
    "a ${unique_token} ${class_token} with a mountain in the background"
    "a ${unique_token} ${class_token} with a blue house in the background"
    "a ${unique_token} ${class_token} on top of a purple rug in a forest"
    "a ${unique_token} ${class_token} with a wheat field in the background"
    "a ${unique_token} ${class_token} with a tree and autumn leaves in the background"
    "a ${unique_token} ${class_token} with the Eiffel Tower in the background"
    "a ${unique_token} ${class_token} floating on top of water"
    "a ${unique_token} ${class_token} floating in an ocean of milk"
    "a ${unique_token} ${class_token} on top of green grass with sunflowers around it"
    "a ${unique_token} ${class_token} on top of a mirror"
    "a ${unique_token} ${class_token} on top of the sidewalk in a crowded street"
    "a ${unique_token} ${class_token} on top of a dirt road"
    "a ${unique_token} ${class_token} on top of a white rug"
    "a red ${unique_token} ${class_token}"
    "a purple ${unique_token} ${class_token}"
    "a shiny ${unique_token} ${class_token}"
    "a wet ${unique_token} ${class_token}"
    "a cube shaped ${unique_token} ${class_token}"
  )

  prompt_test_list=(
    "a ${class_token} in the jungle"
    "a ${class_token} in the snow"
    "a ${class_token} on the beach"
    "a ${class_token} on a cobblestone street"
    "a ${class_token} on top of pink fabric"
    "a ${class_token} on top of a wooden floor"
    "a ${class_token} with a city in the background"
    "a ${class_token} with a mountain in the background"
    "a ${class_token} with a blue house in the background"
    "a ${class_token} on top of a purple rug in a forest"
    "a ${class_token} with a wheat field in the background"
    "a ${class_token} with a tree and autumn leaves in the background"
    "a ${class_token} with the Eiffel Tower in the background"
    "a ${class_token} floating on top of water"
    "a ${class_token} floating in an ocean of milk"
    "a ${class_token} on top of green grass with sunflowers around it"
    "a ${class_token} on top of a mirror"
    "a ${class_token} on top of the sidewalk in a crowded street"
    "a ${class_token} on top of a dirt road"
    "a ${class_token} on top of a white rug"
    "a red ${class_token}"
    "a purple ${class_token}"
    "a shiny ${class_token}"
    "a wet ${class_token}"
    "a cube shaped ${class_token}"
  )

else
  prompt_list=(
    "a ${unique_token} ${class_token} in the jungle"
    "a ${unique_token} ${class_token} in the snow"
    "a ${unique_token} ${class_token} on the beach"
    "a ${unique_token} ${class_token} on a cobblestone street"
    "a ${unique_token} ${class_token} on top of pink fabric"
    "a ${unique_token} ${class_token} on top of a wooden floor"
    "a ${unique_token} ${class_token} with a city in the background"
    "a ${unique_token} ${class_token} with a mountain in the background"
    "a ${unique_token} ${class_token} with a blue house in the background"
    "a ${unique_token} ${class_token} on top of a purple rug in a forest"
    "a ${unique_token} ${class_token} wearing a red hat"
    "a ${unique_token} ${class_token} wearing a santa hat"
    "a ${unique_token} ${class_token} wearing a rainbow scarf"
    "a ${unique_token} ${class_token} wearing a black top hat and a monocle"
    "a ${unique_token} ${class_token} in a chef outfit"
    "a ${unique_token} ${class_token} in a firefighter outfit"
    "a ${unique_token} ${class_token} in a police outfit"
    "a ${unique_token} ${class_token} wearing pink glasses"
    "a ${unique_token} ${class_token} wearing a yellow shirt"
    "a ${unique_token} ${class_token} in a purple wizard outfit"
    "a red ${unique_token} ${class_token}"
    "a purple ${unique_token} ${class_token}"
    "a shiny ${unique_token} ${class_token}"
    "a wet ${unique_token} ${class_token}"
    "a cube shaped ${unique_token} ${class_token}"
  )

  prompt_test_list=(
    "a ${class_token} in the jungle"
    "a ${class_token} in the snow"
    "a ${class_token} on the beach"
    "a ${class_token} on a cobblestone street"
    "a ${class_token} on top of pink fabric"
    "a ${class_token} on top of a wooden floor"
    "a ${class_token} with a city in the background"
    "a ${class_token} with a mountain in the background"
    "a ${class_token} with a blue house in the background"
    "a ${class_token} on top of a purple rug in a forest"
    "a ${class_token} wearing a red hat"
    "a ${class_token} wearing a santa hat"
    "a ${class_token} wearing a rainbow scarf"
    "a ${class_token} wearing a black top hat and a monocle"
    "a ${class_token} in a chef outfit"
    "a ${class_token} in a firefighter outfit"
    "a ${class_token} in a police outfit"
    "a ${class_token} wearing pink glasses"
    "a ${class_token} wearing a yellow shirt"
    "a ${class_token} in a purple wizard outfit"
    "a red ${class_token}"
    "a purple ${class_token}"
    "a shiny ${class_token}"
    "a wet ${class_token}"
    "a cube shaped ${class_token}"
  )
fi


validation_prompt=${prompt_list[$prompt_idx]}
test_prompt=${prompt_test_list[$prompt_idx]}
name="${selected_subject}-${prompt_idx}"
instance_prompt="a photo of ${unique_token} ${class_token}"
class_prompt="a photo of ${class_token}"

export OUTPUT_DIR="log_householder/eps_${eps}_lr_${lr}/l_${l}/${name}"
export INSTANCE_DIR="../data/dreambooth/${selected_subject}"
export CLASS_DIR="data/class_data/${class_token}"

if [ -d "$OUTPUT_DIR" ]; then
    echo "该目录已存在：$OUTPUT_DIR"
    continue
fi

. /home/shen_yuan/miniconda3/etc/profile.d/conda.sh
conda activate oft

# max_train_steps=600 - 1400
# learning_rate=6e-5 

accelerate launch train_dreambooth_householder.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$instance_prompt" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="$class_prompt" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=5000 \
  --learning_rate=$lr \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2005 \
  --validation_prompt="$validation_prompt" \
  --validation_epochs=1 \
  --seed="0" \
  --name="$name" \
  --num_class_images=200 \
  --eps=$eps \
  --l=$l

done
done