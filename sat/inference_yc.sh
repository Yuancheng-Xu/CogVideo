#! /bin/bash
# cd /root/projects/CogVideo/sat; conda deactivate; conda activate cogvideo


##### original model
### need to provide 5b/2b in configs/inference.yaml
# CUDA_VISIBLE_DEVICES=3 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
#  python sample_video.py --base configs/cogvideox_5b.yaml configs/inference.yaml --seed $RANDOM

##### lora finetuned model
### Need to specify the model path, input_file and output path in configs/inference_yc.yaml
# CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
#  python sample_video.py --base configs/cogvideox_5b_lora.yaml configs/inference_yc.yaml --seed $RANDOM

##### original model I2V
# need to change model path, data and output path in configs/inference_yc_I2V.yaml
CUDA_VISIBLE_DEVICES=6 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
 python sample_video.py --base configs/cogvideox_5b_i2v.yaml configs/inference_yc_I2V.yaml --seed $RANDOM

# echo "DONE on `hostname`"