# Video Portrait Editing

Adapted from [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero).


## Setup

1. Tested on Python 3.9, CUDA 11.7
2. Requires torch==1.13.1
3. Install Dependencies from requirements.txt
``` shell
pip install -r requirements.txt
```


--- 



## Inference


To run inference 

for Instruct-Pix2Pix 
```
  python pix2pix.py pix2pix_config.json 
```

for Edge Conditioned ControlNet based approach
```
  python controlnetcanny.py controlnet_config.json
```


---


#### Hyperparameters (Optional)

You can define the following hyperparameters:
* **Motion field strength**:   `motion_field_strength_x` = $\delta_x$  and `motion_field_strength_y` = $\delta_y$ (see our paper, Sect. 3.3.1). Default: `motion_field_strength_x=motion_field_strength_y= 12`.
* $T$ and $T'$ (see our paper, Sect. 3.3.1). Define values `t0` and `t1` in the range `{0,...,50}`. Default: `t0=44`, `t1=47` (DDIM steps). Corresponds to timesteps `881` and `941`, respectively. 
* **Video length**: Define the number of frames `video_length` to be generated. Default: `video_length=8`.


---


### Text-To-Video with Pose Control
To directly call our text-to-video generator with pose control, run this python command:
``` python
prompt = 'an astronaut dancing in outer space'
motion_path = '__assets__/poses_skeleton_gifs/dance1_corr.mp4'
out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.gif"
model.process_controlnet_pose(motion_path, prompt=prompt, save_path=out_path)
```


---


### Text-To-Video with Edge Control
To directly call our text-to-video generator with edge control, run this python command:
``` python
prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = '__assets__/canny_videos_mp4/deer.mp4'
out_path = f'./text2video_edge_guidance_{prompt}.mp4'
model.process_controlnet_canny(video_path, prompt=prompt, save_path=out_path)
```

#### Hyperparameters

- `start_t` (Default: 0): Specifies the starting time in seconds for the video processing. A value of 0 means the processing starts from the beginning of the video.

- `end_t` (Default: -1): Sets the end time in seconds for the video processing. A value of -1 indicates that the processing will continue until the end of the video.

- `out_fps` (Default: -1): Determines the frames per second (fps) for processing the video. A value of -1 fps means that the output video will have original video fps.

- `chunk_size` (Default: 8): Defines the number of frames to be processed at once. A smaller chunk size can reduce memory usage, while a larger size might improve processing speed but requires more memory.

- `low threshold` (Default: 100): Canny Edge detection param.

- `high threshold` (Default: 180): Canny Edge detection param. Make sure that `high_threshold` > `low_threshold`.


---


## Results

### Original


### Video Instruct Pix2Pix
Prompt: "Turn the woman's clothes to superman costume"


### ControlNet with Edge Guidance
Prompt: "Beautiful girl in superman costume in front of white background, a high-quality, detailed, and professional photo"



