import os
import random

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .model import (
    LeafDataset,
    LeafInferenceDataset,
    create_model,
    train_and_validate,
    predict
)



def main():
    # Constants for image dimensions
    # only PADDED values are used
    # HEIGHT = 1400
    # WIDTH = 875

    # Pad imagse as required by UNet11
    PADDED_HEIGHT = 1408
    PADDED_WIDTH = 896

    # setup data directories
    root_directory = os.path.join("../datasets")
    masks_directory = root_directory

    train_images_directory = os.path.join(root_directory, 'train')
    test_images_directory = os.path.join(root_directory, 'test')

    # extract filenames
    images_filenames = sorted(os.listdir(train_images_directory))
    correct_images_filenames = [i for i in images_filenames if cv2.imread(os.path.join(train_images_directory, i)) is not None]

    test_images_filenames = sorted(os.listdir(test_images_directory))
    correct_test_filenames = [i for i in test_images_filenames if cv2.imread(os.path.join(test_images_directory, i)) is not None]

    random.seed(42)
    random.shuffle(correct_images_filenames)

    # split data filenames
    train_images_filenames = correct_images_filenames[0:13]
    val_images_filenames = correct_images_filenames[13:]
    test_images_filenames = correct_test_filenames

    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=PADDED_HEIGHT, min_width=PADDED_WIDTH, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=50, p=0.5),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        strict=True,
        seed=137,
    )
    train_dataset = LeafDataset(train_images_filenames, train_images_directory, masks_directory, transform=train_transform)

    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=PADDED_HEIGHT, min_width=PADDED_WIDTH, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
        strict=True,
        seed=137,
    )
    val_dataset = LeafDataset(val_images_filenames, train_images_directory, masks_directory, transform=val_transform)

    test_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=PADDED_HEIGHT, min_width=PADDED_WIDTH, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
    )
    test_dataset = LeafInferenceDataset(test_images_filenames, test_images_directory, transform=test_transform)


    params = {
        "model": "UNet11",
        "device": "cpu",
        "lr": 0.001,
        "batch_size": 2,
        #"num_workers": 4,
        "epochs": 10,
    }

    model = create_model(params)
    model = train_and_validate(model, train_dataset, val_dataset, params)

    predictions = predict(model, params, test_dataset, batch_size=2)

    predicted_masks = []
    for predicted_padded_mask, original_height, original_width in predictions:
        #cropped_mask = F.center_crop(predicted_padded_mask, original_height, original_width)
        predicted_masks.append(predicted_padded_mask)

    #display_test_image_grid(test_images_filenames, test_images_directory, predicted_masks=predicted_masks)

if __name__ == "__main__":
    main()
