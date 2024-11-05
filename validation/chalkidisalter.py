import logging
import click
import os
import json
import time
import numpy as np
from scipy.special import expit
import tensorflow as tf
from tensorflow_addons.metrics import MeanMetricWrapper
from experiments.model import Classifier, NATIVE_BERT
from experiments.data_loader import SampleGenerator
from data import MODELS_DIR, DATA_DIR
from datasets import load_dataset
from utils.logger import setup_logger
from sklearn.metrics import f1_score, average_precision_score

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log_name = setup_logger()
LOGGER = logging.getLogger(__name__)
cli = click.Group()


@cli.command()
@click.option('--bert_path', default='xlm-roberta-base')
@click.option('--native_bert', default=False)
@click.option('--use_adapters', default=False)
@click.option('--use_ln', default=False)
@click.option('--bottleneck_size', default=256)
@click.option('--n_frozen_layers', default=0)
@click.option('--epochs', default=70)
@click.option('--batch_size', default=8)
@click.option('--learning_rate', default=3e-5)
@click.option('--label_smoothing', default=0.0)
@click.option('--max_document_length', default=512)
@click.option('--monitor', default='val_f1')
@click.option('--train_lang', default='nl')
@click.option('--label_level', default='level_1')
@click.option('--train_samples', default=None)
@click.option('--eval_samples', default=None)
def train(bert_path, native_bert, use_adapters, use_ln, bottleneck_size, n_frozen_layers, epochs, batch_size,
          learning_rate, label_smoothing, monitor, train_lang, label_level,
          max_document_length, train_samples, eval_samples):

    # Simplified setup for monolingual
    eval_langs = [train_lang]  # Use single language for evaluation
    label_index = load_labels(label_level)

    # Load dataset
    train_dataset = load_dataset('multi_eurlex', language=train_lang, label_level=label_level, split='train', trust_remote_code=True)
    eval_dataset = load_dataset('multi_eurlex', language=train_lang, label_level=label_level, trust_remote_code=True)

    # Generators
    train_generator = SampleGenerator(
        dataset=train_dataset[:train_samples if train_samples else len(train_dataset)],
        label_index=label_index,
        lang=train_lang,
        bert_model_path=bert_path, batch_size=batch_size, shuffle=True,
        max_document_length=max_document_length)

    dev_generator = SampleGenerator(
        dataset=eval_dataset['validation'][:eval_samples if eval_samples else len(eval_dataset['validation'])],
        label_index=label_index,
        lang=train_lang,
        bert_model_path=bert_path, batch_size=batch_size, shuffle=False,
        max_document_length=max_document_length)

    # Model setup
    classifier, monitor_metric, monitor_mode = setup_model(bert_path, len(label_index), monitor,
                                                           n_frozen_layers, use_adapters, use_ln,
                                                           bottleneck_size, learning_rate, label_smoothing)
    classifier.summary(print_fn=LOGGER.info, line_length=100)

    # Train model
    history = classifier.fit(train_generator, validation_data=dev_generator,
                             epochs=epochs,
                             callbacks=[tf.keras.callbacks.EarlyStopping(
                                 patience=5, monitor=monitor, mode=monitor_mode, restore_best_weights=True)])

    # Evaluation
    evaluate_model(dev_generator, classifier, label_index, train_lang)

    # Save model
    classifier.save_model(os.path.join(MODELS_DIR, LOGGER.name))


def load_labels(label_level):
    with open(os.path.join(DATA_DIR, 'eurovoc_concepts.json')) as file:
        return {idx: concept for idx, concept in enumerate(json.load(file)[label_level])}


def setup_model(bert_path, num_labels, monitor, n_frozen_layers, use_adapters, use_ln, bottleneck_size, learning_rate, label_smoothing):
    classifier = Classifier(bert_model_path=bert_path, num_labels=num_labels)
    classifier.adapt_model(use_adapters=use_adapters,
                           use_ln=use_ln,
                           bottleneck_size=bottleneck_size,
                           num_frozen_layers=n_frozen_layers)

    if monitor == 'val_f1':
        monitor_metric = 'val_f1'
        monitor_mode = 'max'
    else:
        raise ValueError(f'Monitor "{monitor}" is not supported')

    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=label_smoothing),
                       metrics=[MeanMetricWrapper(f1_score, name="f1_score")])

    return classifier, monitor_metric, monitor_mode


def evaluate_model(generator, classifier, label_index, lang):
    n_documents = len(generator)
    y_true = np.zeros((n_documents, len(label_index)), dtype=np.float32)
    y_pred = np.zeros((n_documents, len(label_index)), dtype=np.float32)

    for i, (x_batch, y_batch) in enumerate(generator):
        yp_batch = classifier.predict(x_batch)
        y_true[i:i+len(yp_batch)] = y_batch
        y_pred[i:i+len(yp_batch)] = expit(yp_batch)  # Apply sigmoid to predictions

    # Calculate F1 and MAP
    f1 = f1_score(y_true, y_pred > 0.5, average='macro')  # Macro F1
    map_score = np.mean([average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(len(label_index))])

    LOGGER.info(f"Language '{lang}' Evaluation - F1 Score: {f1:.4f}, Mean Average Precision (MAP): {map_score:.4f}")


if __name__ == '__main__':
    train()
