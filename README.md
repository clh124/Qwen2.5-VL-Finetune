# Fine-tuning Qwen2.5-VL 


## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yaml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification: 


```json
[
  {
    "video": "ca979ba551c2db3344b0308a54a77da2.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nHow would you judge the engagement continuation rate of the given content, where engagement continuation rate represents the probability of watch time exceeding 5 seconds. The title of the video is None, and the description of the video is None"
      },
      {
        "from": "gpt",
        "value": "28.5064"
      }
    ]
  },
  {
    "video": "61a4b3d62d4c42f0d115cd8c1cf9ed0b.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nHow would you judge the engagement continuation rate of the given content, where engagement continuation rate represents the probability of watch time exceeding 5 seconds. The title of the video is  fatnfurious  sillydilly  smoked , and the description of the video is If it was a race I\u2019d need 2 balers + a beer cooler "
      },
      {
        "from": "gpt",
        "value": "42.0184"
      }
    ]
  },
  ...
]
```


## Training

```bash
bash scripts/finetune_video.sh
```
**Note:** Before running, open scripts/finetune_video.sh and modify the relevant path parameters (e.g., MODEL_NAME, data_path, image_folder, output_dir) to match your own environment.



## Inference


```bash
cd infer

python infer_evqa.py \
  --model-path /path/to/your/model \
  --question-file /path/to/questions.json \
  --save-csv /path/to/save/results.csv
```


## Acknowledgement

This project is based on [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune).