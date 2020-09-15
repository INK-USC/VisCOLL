# COCO-shift and Flickr-shift datasets

## COCO shift
File name format: task_buffer_real_split_{1}_split_{2}_novel_comps_{3}_task_partition_any_objects.pkl

1. The official data split where examples are drawn from
2. How the dataset is applied during training. We separate out 5,000 examples from original train split as the validation examples. The official val split is used as the test split
3. Whether the data is the novel-composition split of 24 held-out concept pairs

The pickled file is python list where each item is a dict:
{'annotation': {'image_id': 374041, 'id': 31929, 'caption': 'Someone sitting on an elephant standing by a fence.  '}, 'task': 'cow', 'mask': (3, 5)}

The `mask` is the text span that should be masked for prediction.  `image_id` can be used for locating the image in the dataset.

## Flicr shift
We use official split for Flickr30k Entities.

{'tokens': ['a', 'motorcyclist', '"', 'pops', 'a', 'wheelie', '"', 'in', 'a', 'grassy', 'field', 'framed', 'by', 'rolling', 'hills', '.'], 'task_id': 518, 'image_id': '3353962769', 'instance_id': 8714, 'phrase_offset': (0, 2)}

The `phrase offset` is the text span that should be masked for prediction. `image_id` can be used for locating the image in the dataset.

## Visualizing the stream
We provide `visual_stream.ipynb` to visualize task distributions over time in Flickr-shift dataset.