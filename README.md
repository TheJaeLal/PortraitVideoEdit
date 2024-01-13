# Video Portrait Editing

## [Click Here for Analysis Report](Analysis_Report.md)


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


Adapted from [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero).



