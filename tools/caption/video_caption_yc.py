# conda deactivate; conda activate cogvideo_caption
import io

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

from matplotlib import pyplot as plt
import pandas as pd

from tqdm import tqdm
import time
import os

MODEL_PATH = "THUDM/cogvlm2-llama3-caption"

PROMPT="Please describe this video in detail." # original prompt
video_csv_path = "/root/projects/Data/CelebV-HQ/video_paths_subset_100_full_path.csv" # provide the path here

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def get_video_length_in_seconds(video_data):
      bridge.set_bridge('torch')
      mp4_stream = video_data
      num_frames = 24
      decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

      frame_id_list = None
      total_frames = len(decord_vr)

      timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
      timestamps = [i[0] for i in timestamps]
      max_second = round(max(timestamps)) + 1

      return max_second


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    device_map = "auto"
).eval()


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    print(video.size())

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    # inputs = {
    #     'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    #     'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    #     'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    #     'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    # }
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).cuda(),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).cuda(),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).cuda(),
        'images': [[inputs['images'][0].cuda().to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 500,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

df = pd.read_csv(video_csv_path)
video_path_list = df["path"].tolist()
caption_list = []

video_seconds_list = []
process_time_list = []

for i, video_path in enumerate(tqdm(video_path_list)):
      temperature = 0.1

      start_time = time.time()
      video_data = open(video_path, 'rb').read()
      response = predict(PROMPT, video_data, temperature)
      end_time = time.time()
      
      caption_list.append(response)

      # statistics
      process_time_list.append(end_time - start_time)
      max_seconds = get_video_length_in_seconds(video_data)
      video_seconds_list.append(max_seconds)

      print(f"{video_path} {max_seconds} seconds\n{response}\n\n")

      # if i == 2:
      #       break

# save
# df["text"] = caption_list
# df.to_csv(new_csv_path, index=False)


video_seconds_list = np.array(video_seconds_list)
process_time_list = np.array(process_time_list)

# print(f'processing time per video seconds:{np.mean(process_time_list/video_seconds_list)}')

print(f'processing time per video seconds:{process_time_list.sum()/video_seconds_list.sum()}')

print(f'total video time:{process_time_list.sum()}')