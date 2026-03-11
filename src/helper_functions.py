import copy
import cv2
import os

import albumentations as A

from collections import defaultdict
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2


class MetricMonitor:
    """
    Aggregate and display stats while training.
    """
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ],
        )


def visualize_augmentations(dataset, idx=0, samples=3):
    """
    Given a dataset, produce images of augmented samples and masks.
    Assumes masks are supplied.
    """
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    _, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


def display_test_image_grid(images_filenames, images_directory, predicted_masks=None):
    """
    Display samples along with their predicated masks.
    """
    cols = 2 if predicted_masks else 1
    rows = len(images_filenames)
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax[i, 0].imshow(image)
        ax[i, 0].set_title("Image")
        ax[i, 0].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 1].imshow(predicted_mask, interpolation="nearest")
            ax[i, 1].set_title("Predicted mask")
            ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()

    