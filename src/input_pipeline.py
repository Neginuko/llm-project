import os
from typing import Dict, List, Optional, Union

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import tokenizer
from clu import deterministic_data
from dataset_preprocessor import DatasetPreprocessor

AUTOTUNE = tf.data.experimental.AUTOTUNE
Features = Dict[str, tf.Tensor]


class NormalizeFeatureNamesOp:
    """Normalizes feature names to 'inputs' and 'targets'."""

    def __init__(self, ds_info: tfds.core.DatasetInfo):
        self.ds_info = ds_info

    def __call__(self, features: Features) -> Features:
        features['inputs'] = features.pop('text')
        # Unnecessary step used for uniformizing with examples/wmt
        features['targets'] = features['inputs']
        return features


def get_raw_dataset(
    dataset_builder: tfds.core.DatasetBuilder, dataset_name: str, split: str
) -> tf.data.Dataset:
    """"""
    num_examples = dataset_builder.info.splits[split].num_examples
    per_host_split = deterministic_data.get_read_instruction_for_host(
        split, num_examples, drop_remainder=False
    )
    ds = dataset_builder.as_dataset(split=per_host_split, shuffle_files=False)

    preprocessor = DatasetPreprocessor()
    ds = preprocessor.preprocess(dataset_name, ds)

    ds = ds.map(
        NormalizeFeatureNamesOp(dataset_builder.info), num_parallel_calls=AUTOTUNE
    )
    return ds


def pack_dataset(
    dataset: tf.data.Dataset,
    key2length: Union[int, Dict[str, int]],
    keys: Optional[List[str]] = None,
):
    """CreTes a 'packed' version of a dataset on-the-fly.

    Adapted from the mesh-tf implementation.

    This is meant to replace the irritation of having to create a separate
    "packed" version of a dataset to train efficiently on TPU.
    Each example in the output dataset represents several examples in the
    input dataset.
    For each key in the input dataset, two additional keys are created:
    <key>_segmentation: an int32 tensor identifying the parts
        representing the original example.
    <key>_position: an int32 tensor identifying the position within the original
        example.
    Example:
    Two input examples get combined to form an output example.
    The input examples are:
    {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
    {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
    The output example is:
    {
                  "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
     "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
         "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                 "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
    "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
        "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
    }
    0 represents padding in both the inputs and the outputs.
    Sequences in the incoming examples are truncated to length "length", and the
    sequences in the output examples all have fixed (padded) length "length".

    Args:
        dataset: a tf.data.Dataset
        key2length: an integer, or a dict from feature-key to integer
        keys: a list of strings (e.g. ["inputs", "targets"])

    Returns:
        a tf.data.Dataset
    """
    shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
    if keys is None:
        keys = list(shapes.keys())
    for k in keys:
        if k not in shapes:
            raise ValueError(
                'Key %s not found in dataset. Available keys are %s'
                % (k, shapes.keys())
            )
        if not shapes[k].is_compatible_with(tf.TensorShape([None])):  # type: ignore[wrong-arg-types]
            raise ValueError('Tensors to be packed must be one-dimensional.')
    # make sure that the length dictionary contains all keys as well as the
    # keys suffixed by "_segmentation" and "_position"
    if isinstance(key2length, int):
        key2length = {k: key2length for k in keys}
    for k in keys:
        for suffix in ['_segmentation', '_position']:
            key2length[k + suffix] = key2length[k]

    # trim length
    dataset = dataset.map(
        lambda x: {k: x[k][: key2length[k]] for k in keys},
        num_parallel_calls=AUTOTUNE,
    )
    # Settings `batch_size=length`` ensures that the concatenated sequences (if they
    # have length >=1) are sufficient to fill at least one packed example.
    batch_size = max(key2length.values())
    dataset = dataset.padded_batch(batch_size, padded_shapes={k: [-1] for k in keys})
    dataset = _pack_with_tf_ops(dataset, keys, key2length)

    # Set the Tensor shapes correctly since they get lost in the process.
    def my_fn(x):
        return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

    return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(
    dataset: tf.data.Dataset, keys: List[str], key2length: Dict[str, int]
) -> tf.data.Dataset:
    pass


def preprocess_data(
    dataset,
    shuffle: bool,
    num_epochs: Optional[int] = 1,
    pack_examples: bool = True,
    shuffle_buffer_size: int = 1024,
    max_length: int = 512,
    batch_size: int = 256,
    drop_remainder: bool = True,
    prefetch_size: int = AUTOTUNE,
):
    """Shuffle and batch/pack the given dataset."""

    def length_filter(max_len):
        def filter_fn(x):
            source, target = x['inputs'], x['targets']
            l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
            return tf.less(l, max_len + 1)

        return filter_fn

    if max_length > 0:
        dataset = dataset.filter(length_filter(max_length))

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(num_epochs)

    if pack_examples:
        dataset = pack_dataset(dataset, max_length)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={'inputs': max_length, 'targets': max_length},
            padding_values={'inputs': 0, 'targets': 0},
            drop_remainder=drop_remainder,
        )

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset


def get_datasets(
    config: ml_collections.ConfigDict,
    *,
    n_devices: int,
    vocab_path: Optional[str] = None,
):
    """Load and return dataset of batched examples for use during training."""
    if vocab_path is None:
        vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

    train_ds_builder = tfds.builder(config.train_dataset_name)
    train_data = get_raw_dataset(train_ds_builder, config.train_dataset_name, 'train')

    if config.eval_dataset_name:
        eval_ds_builder = tfds.builder(config.eval_dataset_name)
    else:
        eval_ds_builder = train_ds_builder
    eval_data = get_raw_dataset(
        eval_ds_builder, config.eval_dataset_name, config.eval_split
    )

    sp_tokenizer = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path,
        vocab_size=config.vocab_size,
        max_corpus_chars=config.max_corpus_chars,
        spm_train_options=config.spm_train_options,
    )
    train_data = train_data.map(
        tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE
    )
    eval_data = eval_data.map(
        tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE
    )

    batch_size = config.per_device_batch_size * n_devices
    if config.eval_per_device_batch_size > 0:
        eval_batch_size = config.eval_per_device_batch_size * n_devices
    else:
        eval_batch_size = batch_size

    train_ds = preprocess_data(
        train_data,
        shuffle=True,
        num_epochs=None,
        pack_examples=True,
        batch_size=batch_size,
        max_length=config.max_target_length,
    )
