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
https://github.com/TheJaeLal/PortraitVideoEdit/assets/24888438/a69c8e30-bf7b-4021-a22b-9d560393161f

### Video Instruct Pix2Pix
Prompt: "Turn the woman's clothes to superman costume"


https://github.com/TheJaeLal/PortraitVideoEdit/assets/24888438/4078f719-50dc-49df-bed1-20890e8b5e0f



### ControlNet with Edge Guidance
Prompt: "Beautiful girl in superman costume in front of white background, a high-quality, detailed, and professional photo"


https://github.com/TheJaeLal/PortraitVideoEdit/assets/24888438/a7f2b9e0-231b-4cd7-a42b-b569eade3517




