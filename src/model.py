import pandas as pd
import cv2
import numpy as np
import os

import ternausnet.models
import torch
import torch.optim
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm

from .helper_functions import MetricMonitor

from .rl_decode import (
    rl_decode,
    rl_encode,
)

#cudnn.benchmark = True


class LeafDataset(Dataset):
    """
    Dataset class for training and validation data.
    """
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory

        self.annotations = pd.read_csv(masks_directory + "/train.csv")
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Assuming the CSV contains two columns: 'id == image_filename' and 'encoded_mask == annotation'
        if image_filename[-4:] == ".jpg":
            image_filename = image_filename[:-4]
        encoded_mask_df = self.annotations[self.annotations['id'] == image_filename]
        if encoded_mask_df.empty:
            raise ValueError(f"No segmentation data found for {image_filename} in the CSV.")

        encoded_mask = encoded_mask_df['annotation'].values[0]
        mask = rl_decode(encoded_mask)

        mask = mask.astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
    

class LeafInferenceDataset(Dataset):
    """
    Dataset class for test data.
    """
    def __init__(self, images_filenames, images_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, original_size


def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for _, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        stream.set_description(
            f"Epoch: {epoch}. Train.      {metric_monitor}",
        )


def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for _, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                f"Epoch: {epoch}. Validation. {metric_monitor}",
            )


def create_model(params):
    model = getattr(ternausnet.models, params["model"])(pretrained=True)
    return model.to(params["device"])


def train_and_validate(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        #num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        #num_workers=params["num_workers"],
        pin_memory=True,
    )
    #criterion = nn.BCEWithLogitsLoss().to(params["device"])
    criterion = (1 - BinaryJaccardIndex()).to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        validate(val_loader, model, criterion, epoch, params)
    return model


def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        #num_workers=params["num_workers"],
        pin_memory=True,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_masks = (probabilities >= 0.5).float() * 1
            predicted_masks = predicted_masks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                predicted_masks,
                original_heights.numpy(),
                original_widths.numpy(),
            ):
                predictions.append((predicted_mask, original_height, original_width))
    return predictions