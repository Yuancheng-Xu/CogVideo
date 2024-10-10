#! /bin/bash
# cd /root/projects/CogVideo/sat; conda deactivate; conda activate cogvideo

### NOTE: we mainly need to modify configs/sft_yc.yaml for batch size and dataset
# if using 2b, need to modify bf16 & fp16 in configs/sft_yc.yaml

# cuda=0,1,2,3,4,5,6,7
cuda=0,1,2,3,4,5,6,7 # probably need to use only 1 or 2 gpus to run the first time to set up the cpu somehow


num_GPU=$(echo $cuda | awk -F, '{print NF}')

### T2V
CUDA_VISIBLE_DEVICES=$cuda \
 torchrun --standalone --nproc_per_node=$num_GPU train_video.py \
 --base configs/cogvideox_5b_lora.yaml configs/sft_yc.yaml --seed $RANDOM 

### I2V
# CUDA_VISIBLE_DEVICES=$cuda \
# torchrun --standalone --nproc_per_node=$num_GPU train_video.py \
# --base configs/cogvideox_5b_i2v_lora.yaml configs/sft_I2V_yc.yaml --seed $RANDOM 



echo "Done Training"