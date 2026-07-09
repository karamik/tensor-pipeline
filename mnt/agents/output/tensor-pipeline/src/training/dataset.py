#!/usr/bin/env python3
"""
Tensor Pipeline — Dataset Loader
Modular data loading for training pipeline.
Supports tf.data.Dataset with caching, prefetching, and augmentation.
"""

import os
import tensorflow as tf

# Configuration
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 10


def create_dummy_dataset(
    num_samples: int = 1000,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Create a dummy dataset for demonstration.
    Replace with your actual data loading logic.
    
    Args:
        num_samples: Number of synthetic samples to generate
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
    
    Returns:
        tf.data.Dataset: Batched and prefetched dataset
    """
    # Generate random synthetic data
    images = tf.random.normal((num_samples, 224, 224, 3), dtype=tf.float32)
    labels = tf.random.uniform((num_samples,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(num_samples, 10000))
    
    if augment:
        dataset = dataset.map(_augment, num_parallel_calls=AUTOTUNE)
    
    # Batch and optimize pipeline
    dataset = (
        dataset
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    
    return dataset


def load_from_directory(
    data_dir: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    image_size: tuple = IMAGE_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load images from directory structure (class folders).
    
    Expected structure:
        data_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img1.jpg
                ...
    
    Args:
        data_dir: Path to dataset root directory
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        image_size: Target image size (height, width)
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    # Optimize performance
    train_ds = _optimize_dataset(train_ds)
    val_ds = _optimize_dataset(val_ds)
    
    return train_ds, val_ds


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random data augmentation."""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def _optimize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply standard optimizations to dataset pipeline."""
    return dataset.cache().prefetch(AUTOTUNE)


def get_dataset_info(dataset: tf.data.Dataset) -> dict:
    """Return information about the dataset."""
    info = {
        "element_spec": str(dataset.element_spec),
        "cardinality": int(dataset.cardinality().numpy()) if dataset.cardinality() != tf.data.INFINITE_CARDINALITY else "infinite",
    }
    return info


if __name__ == "__main__":
    # Test dummy dataset
    print("Testing dummy dataset...")
    ds = create_dummy_dataset(num_samples=100, batch_size=8)
    info = get_dataset_info(ds)
    print(f"Dataset info: {info}")
    
    # Print one batch
    for images, labels in ds.take(1):
        print(f"Batch shape: images={images.shape}, labels={labels.shape}")
        break
    
    print("✅ Dataset loader test passed")
