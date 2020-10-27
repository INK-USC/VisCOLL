# from maskrcnn_benchmark.data.transforms import transforms as T
from torchvision.transforms import transforms as T

def build_transforms(cfg, split="train"):
    if split=="train":
        min_size = min(cfg.EXTERNAL.IMAGE.HEIGHT,cfg.EXTERNAL.IMAGE.WIDTH)
        max_size = max(cfg.EXTERNAL.IMAGE.HEIGHT,cfg.EXTERNAL.IMAGE.WIDTH)
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = min(cfg.EXTERNAL.IMAGE.HEIGHT, cfg.EXTERNAL.IMAGE.WIDTH)
        max_size = max(cfg.EXTERNAL.IMAGE.HEIGHT, cfg.EXTERNAL.IMAGE.WIDTH)
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD,
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform