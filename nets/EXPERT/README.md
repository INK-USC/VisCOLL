`config/mlmcaptioning/config_expert.json:`
Sample configurations for VLBERT model, put into `./` folder (can be edited in `config_mlm.yaml`)
merged with args settings from EXPERT in https://github.com/cvlab-columbia/expert/blob/master/main.py

Pointing is for finding new words from ref. set (please refer to the Learning to Learn paper)

`data/coco.py`
Add tokenizer for VLBERT (condition: EXTERNAL.MLM exists in config file)
output_dict: captions: (text_ids, text_len)

`mlmcaption.py`
require: decoding


## RUN
`mkdir datasets`
`ln -s /home/xsjin/coco datasets/coco`
`mkdir logs`
`CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mlmcaptioning/naive.yaml `

## Requirements:
yacs
opencv-python  
recordclass
nltk
pytorch-transformers
matplotlib