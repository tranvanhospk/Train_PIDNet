import albumentations
import numpy as np




train_albumentations = albumentations.Compose([
    # albumentations.ColorJitter(),
    # albumentations.CLAHE(),
    # albumentations.RandomSnow(),
    # albumentations.RGBShift(),
    # albumentations.RandomShadow(),

    albumentations.HorizontalFlip(),

    albumentations.Affine(rotate=(-10, 10)),
    albumentations.Affine(translate_px={'x': (-200, 200)}),
    albumentations.Affine(translate_px={'y': (-100, 100)}),

    ])

