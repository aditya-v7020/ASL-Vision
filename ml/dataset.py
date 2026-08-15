import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ml.config import (
    TRAIN_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    MAX_IMAGES_PER_CLASS,
    CLASSES,
    NUM_CLASSES,
)

def get_image_file_list(train_dir=TRAIN_DIR, max_per_class=MAX_IMAGES_PER_CLASS, seed=RANDOM_SEED):
    """
    Scans the training directory and builds stratified lists of file paths and labels
    for Training, Validation, and Test sets with zero data leakage.
    Uses the authoritative CLASSES list from ml.config.
    """
    random.seed(seed)
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    class_names = CLASSES
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_stats = {}

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        all_images = [
            os.path.join(class_dir, f)
            for f in sorted(os.listdir(class_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Apply reproducible shuffle per class before slicing
        rng = random.Random(seed + class_to_idx[class_name])
        rng.shuffle(all_images)
        
        if max_per_class is not None and max_per_class > 0:
            selected_images = all_images[:max_per_class]
        else:
            selected_images = all_images
            
        n_total = len(selected_images)
        n_train = int(n_total * TRAIN_SPLIT)
        n_val = int(n_total * VAL_SPLIT)
        
        c_train = selected_images[:n_train]
        c_val = selected_images[n_train:n_train + n_val]
        c_test = selected_images[n_train + n_val:]
        
        idx = class_to_idx[class_name]
        
        train_paths.extend(c_train)
        train_labels.extend([idx] * len(c_train))
        
        val_paths.extend(c_val)
        val_labels.extend([idx] * len(c_val))
        
        test_paths.extend(c_test)
        test_labels.extend([idx] * len(c_test))
        
        class_stats[class_name] = {
            "total": n_total,
            "train": len(c_train),
            "val": len(c_val),
            "test": len(c_test),
        }
        
    # Verify zero data overlap
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)
    
    assert len(train_set.intersection(val_set)) == 0, "Data leakage between Train and Validation sets!"
    assert len(train_set.intersection(test_set)) == 0, "Data leakage between Train and Test sets!"
    assert len(val_set.intersection(test_set)) == 0, "Data leakage between Validation and Test sets!"
    
    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        class_names,
        class_stats
    )


def load_and_preprocess_image(path, label):
    """
    Decodes an image file from path, converts to RGB, and resizes to 128x128.
    """
    img_raw = tf.io.read_file(path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return img, label


def get_data_augmentation_pipeline():
    """
    Creates a data augmentation model tailored for ASL gesture recognition.
    Applies subtle rotation (+/- 5%), zoom (+/- 5%), translation (+/- 5%), and contrast (+/- 10%).
    Horizontal flip is strictly omitted to preserve asymmetric ASL letters (e.g. J, Z, G, H, P, Q).
    """
    return keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def create_dataset_pipeline(file_paths, labels, is_training=False, batch_size=BATCH_SIZE):
    """
    Builds a high-throughput tf.data pipeline with parallel mapping, vector batching, and prefetching.
    """
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(file_paths), 10000), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    
    if is_training:
        aug = get_data_augmentation_pipeline()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_datasets(max_per_class=MAX_IMAGES_PER_CLASS, batch_size=BATCH_SIZE):
    """
    Helper to generate train, val, and test tf.data.Dataset objects.
    """
    (train_data, val_data, test_data, class_names, stats) = get_image_file_list(
        max_per_class=max_per_class
    )
    
    train_ds = create_dataset_pipeline(train_data[0], train_data[1], is_training=True, batch_size=batch_size)
    val_ds = create_dataset_pipeline(val_data[0], val_data[1], is_training=False, batch_size=batch_size)
    test_ds = create_dataset_pipeline(test_data[0], test_data[1], is_training=False, batch_size=batch_size)
    
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_names": class_names,
        "stats": stats,
        "counts": {
            "train": len(train_data[0]),
            "val": len(val_data[0]),
            "test": len(test_data[0]),
        },
        "raw_test_data": test_data,
    }
