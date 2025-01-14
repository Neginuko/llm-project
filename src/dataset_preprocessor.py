import tensorflow as tf
from absl import logging


class BasePreprocessor:
    def preprocess(self, ds):
        raise NotImplementedError(
            '`preprocess` method must be implemented by subclasses'
        )

    def head(self, ds, n=5):
        for x in ds.take(n):
            logging.info(tf.compat.as_text(x['text'].numpy()))


class Wiki40bJaPreprocessor(BasePreprocessor):
    def preprocess(self, ds):
        MIN_LENGTH = 10

        def extract_text_only(x):
            return {'text': x['text']}

        def split_paragraph_py(x):
            text = x.numpy().decode('utf-8')
            lines = text.split('\n')
            paragraphs = []

            for i, line in enumerate(lines):
                if '_START_PARAGRAPH_' in line:
                    next_line_index = i + 1
                    if next_line_index < len(line):
                        next_line = lines[next_line_index]
                        paragraphs.extend(next_line.split('_NEWLINE_'))

            if not paragraphs:
                paragraphs.append('')
            return [paragraphs]

        def split_paragraph(x):
            text = x['text']
            parts = tf.py_function(split_paragraph_py, [text], Tout=[tf.string])
            parts.set_shape([None])
            return tf.data.Dataset.from_tensor_slices({'text': parts})

        def remove_short(x):
            return tf.strings.length(x['text'], unit='UTF8_CHAR') >= MIN_LENGTH

        return ds.map(extract_text_only).flat_map(split_paragraph).filter(remove_short)


class Cc100JaPreprocessor(BasePreprocessor):
    def preprocess(self, ds):
        def extract_text_only(x):
            return {'text': x['text']}

        def remove_newlines(x):
            x['text'] = tf.strings.regex_replace(x['text'], '\n', '')
            return x

        def remove_empty(x):
            return tf.strings.length(x['text']) > 0

        return ds.map(extract_text_only).map(remove_newlines).filter(remove_empty)


class Lm1bPreprocessor(BasePreprocessor):
    def preprocess(self, ds):
        return ds


class DatasetPreprocessor:
    def __init__(self):
        self.preprocessor = {
            'lm1b': Lm1bPreprocessor(),
            'wiki40b/ja': Wiki40bJaPreprocessor(),
            'huggingface/cc100/ja': Cc100JaPreprocessor(),
        }

    def preprocess(self, dataset_name: str, ds: tf.data.Dataset):
        if dataset_name not in self.preprocessor:
            logging.warning('No preprocessor found for %s', dataset_name)
            return ds

        preprocessor = self.preprocessor[dataset_name]
        result = preprocessor.preprocess(ds)
        preprocessor.head(result, n=10)
        return result
