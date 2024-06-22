"""Tokenizer for datasets encoding."""

import dataclasses
import os
import tempfile
import time
from typing import Any, Dict, Iterable, Optional

import jax
import sentencepiece as spm
import tensorflow as tf
from absl import logging

Features = Dict[str, tf.Tensor]


def _dump_chars_to_textfile(
    dataset: tf.data.Dataset,
    *,
    max_chars: int = int(1e7),
    data_keys=('inputs', 'targets'),
):
    char_count = 0
    ds_iter = dataset.as_numpy_iterator()
    with tempfile.NamedTemporaryFile(delete=False, prefix='/tmp/ds_chars') as outfp:
        while char_count < max_chars:
            example = next(ds_iter)
            for k in data_keys:
                line = example[k] + b'\n'
                char_count += len(line)
                outfp.write(line)
    return outfp.name, char_count


def _train_sentencepiece(
    dataset,
    *,
    vocab_size: int,
    max_chars: int = int(1e7),
    model_path: str,
    model_type: str = 'unigram',
    spm_train_options: Optional[Dict[str, Any]] = None,
    data_keys=('inputs', 'targets'),
):
    """Train SentencePiece tokenizer from given dataset."""
    abs_model_path = os.path.abspath(model_path)
    fname, _ = _dump_chars_to_textfile(
        dataset, max_chars=max_chars, data_keys=data_keys
    )
    with tempfile.NamedTemporaryFile(delete=False, prefix='/tmp/sp_tmp') as model_fp:
        pass  # just gets the prefix'd temp-filename
    spm.SentencePieceTrainer.Train(
        input=fname,
        model_prefix=model_fp.name,
        vocab_size=vocab_size,
        model_type=model_type,
        **spm_train_options,
    )
    if jax.process_index() == 0:
        copy_rename_path = abs_model_path + '.rntmp'
        tf.io.gfile.copy(model_fp.name + '.model', copy_rename_path, overwrite=True)
        tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)
        logging.info('copied %s to %s', model_fp.name + '.model', abs_model_path)
    else:
        while not tf.io.gfile.exists(abs_model_path):
            time.sleep(1)
        time.sleep(1)
    return abs_model_path


def _load_sentencepiece_tokenizer(
    model_path: str, add_bos: bool = False, add_eos: bool = True, reverse: bool = False
):
    """Load SentencePiece tokenizer from given model filepath."""
    sp = spm.SentencePieceProcessor()
    sp.Init(model_file=model_path, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
    return sp


def load_or_train_tokenizer(
    dataset: tf.data.Dataset,
    *,
    vocab_path: str,
    vocab_size: int,
    max_corpus_chars: int = int(1e7),
    spm_train_options: Optional[Dict[str, Any]] = None,
    data_keys=('inputs', 'targets'),
):
    """Load the tokenizer at `vocab_path` or trains a one from `dataset`."""
    try:
        _load_sentencepiece_tokenizer(vocab_path)
    except OSError:
        # * google/sentencepiece python wrapped lib throws OSError
        # * when model file is not found.
        logging.info('SentencePiece model not found, building it from dataset...')
        vocab_path = _train_sentencepiece(
            dataset,
            vocab_size=vocab_size,
            max_chars=max_corpus_chars,
            model_path=vocab_path,
            spm_train_options=spm_train_options,
            data_keys=data_keys,
        )
        return _load_sentencepiece_tokenizer(vocab_path)


@dataclasses.dataclass
class TokenizeOp:
    sp_tokenizer: Any
    data_keys: Iterable[str] = ('inputs', 'targets')

    def __call__(self, features: Features) -> Features:
        for k in self.data_keys:
            features[k] = self.sp_tokenizer.Encode(features[k])
        return features
