# Infected Tomato Leaf - Vein Segmentation

**This repository is an active work in progress. Expect consistent changes!**

Given a limited dataset of only 27 images of tomato leafs, many of which
display signs of disease at various stages, the goal is to develop a model to
segment the primary and secondary veins. Along with each image, we are given a
binary mask, i.e., a matrix such that each entry corresponds to a pixel, with
the entry being 1 if the corresponding pixel is determined to be on a vein and
0 otherwise. These masks are determined by hand. The problem is difficult due
to having a small dataset and is compounded by the potential presence of
disease, appearing as large dark spots on the leaf. These spots inhibit the use
of paradigms such as edge detection models. To address these difficulties, one
method is to perform data-augmentation via the albumentation library then
training with the nnU-Net framework, a powerful open-source, self-configuring
framework developed for medical image segmentation.

This problem and dataset was part of a kaggle competition which can be found
via the
(link)(https://www.kaggle.com/competitions/infected-tomato-leaf-vein-segmentation/data).
This repository is not associated with any competition submission.

## Segmentation Mask Encoding

Every mask is the same size as the input images (875x1400 pixels) and is
binary, with 0 marking non-vein and 1 marking vein.

### Run-length Encoding

Run-length encoding is a simple compression technique that replaces consecutive
occurrences of a symbol with a count of how many times that symbol occurred
along with just one copy of that symbol.

For example:

```bash
00011110000111000000
```

Would be encoded as:

```bash
3 0 4 1 4 0 3 1 6 0
```

### Run-length Decoding

The code snippet to decode an encoding string into numpy array is:

```python
import numpy as np
HEIGHT = 1400
WIDTH = 875

def rl_decode(enc):
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i+1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape((HEIGHT, WIDTH))
```

### Files

- **train.csv** - the training set of 17 leaf images and their vein segmentation mask
- **test.csv** - the test set of 10 leaf images

### Directories

- **train** - a directory containing the training set leaf images
- **test** - a directory containing the test set leaf images

### Columns

- `id` - the id corresponds with the name of the leaf image (e.g. leaf02
  corresponds with image leaf02.jpg)
- `annotation` -  a flattened and run-length encoded version of the binary vein
  mask for the corresponding image

## Basic Model - Data Augmentation with Albumentations

`notebooks/pytorch_model.ipynb` provides a model which utilizes the
albumentations python library for data augmentation. When an image is chosen
for training, it is first 'augmented' via a sequence of prescribed
    transformations, each being applied with a prescribed probability. Any
    transformation applied to the image is also applied to its mask.
    Transformations may include random rotations, flipping the image, changing
    the image contrast, shifting RGB values, etc. This artificially increases
    the number of test examples by creating slightly modified copies of
    existing data. We use the pretrained UNet11 model provided by the
    TernausNet library `ternausnet.models` and a Jaccard loss function, which
    is discussed below.

### Jaccard Index

The Jaccard index, defined as $J(A,B) = \frac{\vert A\cap B\vert}{\vert A\cup
B\vert}$, is a measure of similarity of two sets $A$ and $B$. It is the ratio
of common elements normalized by the total number of elements in either set. In
the context of a binary segmentation problem, it is the number of entries
correctly predicted by the model normalized by the total of number entries
either in the ground truth or in the prediction. In a perfect prediction, we
have $A \cap B = A \cup B$ and so the Jaccard index would be 1. If our
prediction is a proper superset of the ground truth, the numerator is maximal,
however, the union in the denominator is larger and so the index is less than 1.
We see that we are penalized by improperly assigning entries which are not in
the ground truth. The Jaccard loss is simply $L(A,B) = 1 - J(A,B)$. We use the
`BinaryJaccardIndex` function provided in by TorchMetrics library
`torchmetrics.classification`. It may be necessary to force `loss.requires_grad
= True` before backpropagating.

### Dice Loss

It may be worth investigating the dice loss function $$DL(A,B) = 1 -
\frac{2\vert A \cap B \vert}{\vert A \vert + \vert B \vert}$$ We do not use
BinaryCrossEntropy due to the target segmentation being small and nonlocalized.
