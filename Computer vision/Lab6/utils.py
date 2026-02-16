import torch
import numpy as np
from datasets import load_dataset


def download(data_name):

    # Download the Tiny ImageNet dataset from HuggingFace
    dataset = load_dataset(data_name)
    train_data = dataset['train']
    test_data = dataset['valid']

    return train_data, test_data


def stratified_sampler(images, labels, ratio):
    sampled_images, sampled_labels = [], []

    for label in np.unique(labels):
        mask = np.where(labels == label, True, False)
        sample_idxs = np.random.choice(
            mask.sum(),
            size=int(mask.sum() * ratio),
            replace=False
        )

        sampled_images.append(images[mask][sample_idxs])
        sampled_labels.append(labels[mask][sample_idxs])

    return np.vstack(sampled_images), np.hstack(sampled_labels)


def preprocess(data, sample, ratio):
    images, labels = [], []
    for idx in range(data.shape[0]):

        # Transform each RGB image into a (64x64x3) array
        image_array = np.array(data[idx]['image']).astype(np.float32)
        
        # Scale images by dividing by 255
        image_array /= 255.0

        image_label = data[idx]['label']

        # Exclude gray images with shape (64x64)
        if len(image_array.shape) != 3:
            continue

        labels.append(image_label)
        images.append(image_array)
        
    labels = np.array(labels)
    images = np.array(images)

    if sample and ratio is not None:
        images, labels = stratified_sampler(images, labels, ratio)

    # Reshape images from (Batch, Height, Width, Channel) to (Batch, Channel, Height, Width)
    images = np.transpose(images, (0, 3, 1, 2))
    
    return torch.tensor(images, dtype=torch.float32), torch.LongTensor(labels)


def read_data(data_name, sample=False, ratio=None):
    train_data, test_data = download(data_name)
    train_images, train_labels = preprocess(train_data, sample, ratio)
    test_images, test_labels = preprocess(test_data, sample, ratio)

    print(f'Train images shape: {train_images.size()}, Train labels shape: {train_labels.size()}')
    print(f'Test images shape: {test_images.size()}, Test labels shape: {test_labels.size()}')

    return train_images, test_images, train_labels, test_labels