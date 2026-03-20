import argparse
import os
import random

import cv2
import albumentations as A
import albumentations.augmentations.crops.functional as fcrops
from albumentations.pytorch import ToTensorV2

from dataset_classes import (
    LeafDataset,
    LeafInferenceDataset,
    create_model,
    train_and_validate,
    predict,
)

from helper_functions import (
    display_test_image_grid,
    save_prediction_images,
)


def main(args):
    # Constants for image dimensions
    #HEIGHT = 1400
    #WIDTH = 875

    # Pad imagse as required by UNet11
    PADDED_HEIGHT = 1408
    PADDED_WIDTH = 896

    # setup data directories
    root_directory = os.path.join('datasets')
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
            A.RandomCrop(256, 256),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=50, p=0.5),
            A.Affine(translate_percent=0.2, scale=0.2, rotate=50, p=0.5),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
            #A.OneOf(
            #    [
            #        A.MedianBlur(blur_limit=2, p=0.1),
            #        A.Blur(blur_limit=2, p=0.1),
            #    ],
            #    p=0.5,
            #),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                ],
                p=0.5,
            ),
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


    #params = {
    #    "model": "UNet11",
    #    "device": "mps",
    #    "lr": 0.001,
    #    "batch_size": 2,
    #    #"num_workers": 4,
    #    "epochs": 3,
    #}

    params = {
        "model": args.model,
        "device": args.device,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss": args.loss_function,
    }

    model = create_model(params)
    model = train_and_validate(model, train_dataset, val_dataset, params)

    predictions = predict(model, params, test_dataset, batch_size=2)

    predicted_masks = []
    for predicted_padded_mask, original_height, original_width in predictions:
        crop_coordinates = fcrops.get_center_crop_coords((PADDED_HEIGHT, PADDED_WIDTH),
                                                         (original_height, original_width))
        cropped_mask = fcrops.crop(predicted_padded_mask, *crop_coordinates)
        predicted_masks.append(cropped_mask)

    # set prediction directory
    root_prediction_images_directory = os.path.join(root_directory, 'predictions')
    experiment_prediction_images_directory = os.path.join(root_prediction_images_directory,
                                                          args.experiment_name)

    if not os.path.exists(root_prediction_images_directory):
        os.makedirs(root_prediction_images_directory)

    if not os.path.exists(experiment_prediction_images_directory):
        os.makedirs(experiment_prediction_images_directory)

    save_prediction_images(test_images_filenames, experiment_prediction_images_directory, predicted_masks)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Parameter settings for training')

    # Add arguments
    parser.add_argument('--model', type=str, default='UNet11', choices=['UNet11'], help='model to run: UNet11 hardcoded and is the only model available')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'cpu', 'mps'], help='device to trian on: cuda, cpu, or mps')

    parser.add_argument('--loss_function', type=str, 
                        default='SoftJaccard', 
                        choices=[
                            'SoftJaccard', 
                            'SoftJaccardBCE',
                            'SoftDice',
                            'SoftDiceBCE',
                            'BCE',
                            ], 
                        help='either Soft Jaccard Loss or Soft Dice Loss',
    )
    parser.add_argument('--optimizer', type=str, default="adam", choices=['adam'], help='optimizer to use')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

    parser.add_argument('--train_val_seed', type=int, default=42, help='seed used to split training and validation data')
    parser.add_argument('--augmentation_seed', type=int, default=137, help='seed used for augmenting samples')

    parser.add_argument('--experiment_name', type=str, default='experiment_predictions', help='experiment_name')
    parser.add_argument('--root_data_directory', type=str, default='datasets', help='the root directory the training and test data exists in')

    # Parse the arguments
    args = parser.parse_args()

    main(args)
