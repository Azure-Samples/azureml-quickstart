

# echo "Cloning LLaVa"
# git clone https://github.com/haotian-liu/LLaVA.git

DATA_PATH=$1
OUT_DIR=$2

# echo "~~~~~~~~~~~~~~~~ Creating conda env::::: "

# conda clean --all -y

# conda config --append channels conda-forge

# conda clean --all -y

# conda create -n llava python=3.10 -y

# echo "~~~~~~~~~~~~~~~~~~ Init conda bash env...."
# eval "$(conda shell.bash hook)"

echo "~~~~~~~~~~~~~~~~~~~~~~~~ Activate conda env"
#source activate llava

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
conda env list

# cd LLaVA 

# pip install --upgrade pip  # enable PEP 660 support
# pip install -e .


# # echo "DataPath: $DATA_PATH Out_dir: $OUT_DIR"

# echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
# ls -a 
# echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"


# # # echo "Installing [train] requirements:"
# pip install -e ".[train]"
# # # echo "Installing flash-attn:"
# pip install flash-attn --no-build-isolation

# echo "Install requirements!!!!"

# cd ..




deepspeed /LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /LLaVA/scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path $DATA_PATH/train/dataset.json \
    --image_folder $DATA_PATH/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir $OUT_DIR/checkpoints/llava-v1.5-13b-task-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True 
