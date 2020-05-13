# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
# BASELINE MODEL
#########################################################


class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.batch = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(12)

        self.fc = nn.Linear(8*8*24, 128)
        self.batch3 = nn.BatchNorm2d(128)
        self.cls = nn.Linear(128, 16)

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = F.relu(self.batch(self.conv1(x)))
        x = F.relu(self.batch1(self.conv2(x)))
        x = F.relu(self.batch2(self.conv3(x)))

        x = x.view(-1, 1536)
        x = F.relu(self.batch2(self.fc(x)))
        x = self.cls(x)

        # TODO-BLOCK-END
        return x


def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    optimizer.zero_grad()

    predictions = net.forward(inputs)
    _, highest = torch.max(predictions, 1)
    num_correct = torch.sum(highest == labels.data.reshape(-1))
    loss = criterion(predictions, labels.squeeze())
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN

    loss.backward()
    loss = loss.item()
    running_loss = loss

    optimizer.step()
    total_images = labels.size()[0]

    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
# DATA AUGMENTATION
#########################################################


class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """

    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        x_shift, y_shift = np.random.randint(-1 *
                                             self.max_shift, self.max_shift + 1, 2)

        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        for index in range(3):
            image[index] = cv2.warpAffine(image[index], M, (W, H))
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        contrast = np.random.uniform(
            low=self.min_contrast, high=self.max_contrast)

        mean = np.mean(image)
        if contrast == 0:
            image = image * 0 + mean
        else:
            image = contrast * image
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """

    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        angle = np.random.randint(-1 * self.max_angle, self.max_angle + 1)
        M = cv2.getRotationMatrix2D(((W-1)/2.0, (H-1)/2.0), angle, 1)

        for index in range(3):
            image[index] = cv2.warpAffine(image[index], M, (W, H))
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        if random.random() <= self.p:
            image[0] = cv2.flip(image[0], 1)
            image[1] = cv2.flip(image[1], 1)
            image[2] = cv2.flip(image[2], 1)

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
# STUDENT MODEL
#########################################################


def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds = [54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        Shift(max_shift=6),
        Contrast(min_contrast=0.3, max_contrast=0.9),
        Rotate(max_angle=11),
        HorizontalFlip(p=0.5),
        transforms.Normalize(dataset_means, dataset_stds)
    ])

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN
    batch_size = 128
    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN
    epochs = 60
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer


class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 12, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 36, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 108, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(108)
        self.conv4 = nn.Conv2d(108, 216, 3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm2d(216)
        # self.conv5 = nn.Conv2d(243, 729, 3, stride=2, padding=1)
        # self.batch5 = nn.BatchNorm2d(729)

        self.fc = nn.Linear(216*4, 128)
        # self.fc2 = nn.Linear(500, 128)
        self.drop = nn.Dropout(p=0.5)
        self.cls = nn.Linear(128, 16)

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        # x = F.relu(self.conv4(x))
        x = F.relu(self.batch4(self.conv4(x)))

        x = F.avg_pool2d(x, 2, 2)
        x = x.view(-1, 216*4)
        x = F.relu(self.fc(x))
        # x = self.fc2(x)
        # x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.cls(x)

        # TODO-BLOCK-END
        return x

#########################################################
# ADVERSARIAL IMAGES
#########################################################


def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    return perturbed_image, noise
