# 用于训练VAD
import os
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
# import modal
import numpy as np
import seaborn as sns
# import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import Wav2Vec2Processor
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from datasets import load_dataset, concatenate_datasets, load_from_disk
from logger import log, log_model_structure, log_dataset_statistics, log_dependencies, ProgressLoggerCallback
from model import Wav2Vec2ForEndpointing

DATA_PATH = "**YOUR_DATA_PATH**"
SAVE_PATH = "**YOUR_SAVE_PATH**"

# Hyperparameters and configuration
CONFIG = {
    "model_name": "facebook/wav2vec2-base-960h",

    # Three types of dataset are used during in this script: training, eval, and test.
    #
    # - The eval set is used to guide the training process, for example with early stopping, and selecting
    #   the best checkpoint.
    #
    # - The test set is kept completely separate from the training process, and is periodically used to
    #   evaluate the performance of the model.
    #
    # The datasets in `datasets_training` are split 80/10/10, and used for all three purposes.
    # The datasets in `datasets_test` are only used for testing, and are not split.
    #
    # All test datasets are stored and reported separately.
    "datasets_training": [
       f"{DATA_PATH}/final_audio_ds",
    ],
    "datasets_test": [], # e.g. "/data/datasets/human_5_filler"

    # Training parameters
    "learning_rate": 5e-5,
    "num_epochs": 40,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    # Evaluation parameters
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,
}

def load_dataset_at(path: str):
    # Ignore linter errors, this works fine
    if path.startswith('/'):
        ds = load_from_disk(path)
        return ds.shuffle(seed=947)
    else:
        ds = load_dataset(path)
        return ds.shuffle(seed=947)

def load_audio(file_path: str):
    # Load audio file and return the audio array
    audio_array, _ = librosa.load(file_path, sr=16000)
    return audio_array

def validate_audio_lengths(dataset, dataset_name, dataset_path):
    """Validate that all audio samples are between 0 and 16 seconds"""
    for i, sample in enumerate(dataset):
        audio_array = load_audio(f"{dataset_path}/audio_data/{sample['audio']}")
        duration = len(audio_array) / 16000

        if duration <= 0:
            raise ValueError(
                f"Fatal error: Audio sample {i} in dataset '{dataset_name}' has zero or negative length ({duration} seconds)")

        if duration > 16:
            raise ValueError(
                f"Fatal error: Audio sample {i} in dataset '{dataset_name}' exceeds 16 seconds limit ({duration} seconds)")

def prepare_datasets(preprocess_function, config):
    """
    Loads, splits, and organizes datasets based on config settings.

    Returns a dictionary with "training", "eval", and "test" entries.
    """
    datasets_training = config["datasets_training"]
    datasets_test = config["datasets_test"]

    overlap = set(datasets_training).intersection(set(datasets_test))
    if overlap:
        raise ValueError(f"Found overlapping datasets in training and test: {overlap}")

    training_splits = []
    eval_splits = []
    test_splits = {}

    for dataset_path in datasets_training:
        # Extract dataset name from path
        dataset_name = dataset_path.split("/")[-1]

        full_dataset = load_dataset_at(dataset_path)

        validate_audio_lengths(full_dataset, dataset_name, dataset_path)

        # Create train/eval/test split (80/10/10)
        dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
        training_splits.append(dataset_dict["train"])
        eval_test_dict = dataset_dict["test"].train_test_split(test_size=0.5, seed=42)

        eval_splits.append(eval_test_dict["train"])
        test_splits[dataset_name] = eval_test_dict["test"]

    # Merge training and eval splits
    merged_training_dataset = concatenate_datasets(training_splits).shuffle(seed=42)
    merged_eval_dataset = concatenate_datasets(eval_splits)

    # Load and add the full test datasets
    for dataset_path in datasets_test:
        dataset_name = dataset_path.split("/")[-1]
        test_dataset = load_dataset_at(dataset_path)

        validate_audio_lengths(test_dataset, dataset_name)

        test_splits[dataset_name] = test_dataset

    def apply_preprocessing(dataset):
        return dataset.map(
            preprocess_function,
            batched=True,
            batch_size=8,
            remove_columns=["audio", "endpoint_bool"],
            num_proc=16
        )

    merged_training_dataset = apply_preprocessing(merged_training_dataset)
    merged_eval_dataset = apply_preprocessing(merged_eval_dataset)

    for dataset_name, dataset in test_splits.items():
        test_splits[dataset_name] = apply_preprocessing(dataset)

    return {
        "training": merged_training_dataset,
        "eval": merged_eval_dataset,
        "test": test_splits
    }

def process_predictions(logits):
    """
    Converts raw logits into squeezed probability predictions and binary predictions.
    """
    if np.isnan(logits).any() or not np.isfinite(logits).all():
        raise ValueError("Non-finite or NaN values detected in logits during processing")
    
    probs = logits.squeeze()
    preds = (probs > 0.5).astype(int)
    
    return probs, preds

def get_predictions_and_labels(trainer, dataset, metric_key_prefix=None):
    """
    Returns tuple:
        - predictions: Raw prediction output from trainer
        - labels: Ground truth labels
        - probs: Squeezed probability predictions
        - preds: Binary predictions (probs > 0.5)
    """
    predictions = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)
    
    probs, preds = process_predictions(predictions.predictions)
    labels = predictions.label_ids
    
    return predictions, labels, probs, preds

def evaluate_and_plot(trainer, dataset, split_name):
    log.info(f"Evaluating on {split_name} set...")
    metrics = trainer.evaluate(eval_dataset=dataset)

    predictions, labels, probs, preds = get_predictions_and_labels(trainer, dataset)

    output_dir = os.path.join(trainer.args.output_dir, "evaluation_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    try:
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Incomplete', 'Complete'],
                    yticklabels=['Incomplete', 'Complete'])
        plt.title(f'Confusion Matrix - {split_name.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{split_name}.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        log.info(f"Saved confusion matrix to {confusion_matrix_path}")
    except Exception as e:
        log.error(f"Could not create confusion matrix for {split_name}: {e}")
        confusion_matrix_path = None

    # Plot and save probability distribution
    plt.figure(figsize=(10, 6))
    try:
        plt.hist(probs, bins=50, alpha=0.5, label='Probability of Complete')
        plt.title(f'Distribution of Completion Probabilities - {split_name.capitalize()} Set')
        plt.xlabel('Probability of Complete')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        prob_dist_path = os.path.join(output_dir, f'probability_distribution_{split_name}.png')
        plt.savefig(prob_dist_path)
        plt.close()
        log.info(f"Saved probability distribution to {prob_dist_path}")
    except Exception as e:
        log.error(f"Could not create probability distribution for {split_name}: {e}")
        prob_dist_path = None

    return metrics, predictions

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs, preds = process_predictions(logits)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division="warn"),
        "recall": recall_score(labels, preds, zero_division="warn"),
        "f1": f1_score(labels, preds, zero_division="warn"),
        "pred_positives": tp + fp,
        "pred_negatives": tn + fn,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }

def training_run(run_number):

    log_dependencies()

    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    CONFIG["run_name"] = f"ji_smart_turn-{now}_run{run_number}"

    log.info(f"Starting training run: {CONFIG['run_name']}")

    model = Wav2Vec2ForEndpointing.from_pretrained(CONFIG["model_name"], num_labels=1)
    processor = Wav2Vec2Processor.from_pretrained(CONFIG["model_name"])

    log_model_structure(model, CONFIG)

    dataset_path = CONFIG["datasets_training"][0]

    def preprocess_function(batch):
        audio_arrays = [load_audio(f"{dataset_path}/audio_data/{x}") for x in batch["audio"]]
        labels = [1 if lb else 0 for lb in batch["endpoint_bool"]]

        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,
            return_attention_mask=True,
            return_tensors="pt"
        )
        inputs["labels"] = labels

        inputs["language"] = batch["language"] if "language" in batch else (["cn"] * len(labels))
        if "midfiller" in batch:
            inputs["midfiller"] = batch["midfiller"]
        if "endfiller" in batch:
            inputs["endfiller"] = batch["endfiller"]
        
        return inputs

    datasets = prepare_datasets(preprocess_function, CONFIG)

    log_dataset_statistics("training", datasets["training"])
    log_dataset_statistics("eval", datasets["eval"])

    for dataset_name, dataset in datasets["test"].items():
        log_dataset_statistics("test_" + dataset_name, dataset)

    training_args = TrainingArguments(
        output_dir=f"{SAVE_PATH}/{CONFIG['run_name']}",
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        num_train_epochs=CONFIG["num_epochs"],
        eval_strategy=IntervalStrategy.STEPS,
        gradient_accumulation_steps=1,
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        logging_steps=CONFIG["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="cosine",
        report_to=[],
        max_grad_norm=1.0,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        tf32=True,
        disable_tqdm=True,
    )

    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets["training"],
        eval_dataset=datasets["eval"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            ProgressLoggerCallback(log_interval=CONFIG["logging_steps"])
        ]
    )

    trainer.train()

    # Evaluate on validation set
    log.info(f"Final eval set evaluation:")
    evaluate_and_plot(trainer, datasets["eval"], "eval")

    # Evaluate on test set
    for dataset_name, dataset in datasets["test"].items():
        log.info(f"Test set evaluation ({dataset_name}):")
        evaluate_and_plot(trainer, dataset, dataset_name)

    # Save the final model and processor.
    final_save_path = f"{training_args.output_dir}/final_model"
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    log.info(f"Model saved to {final_save_path}")

def main():
    print("running!")
    training_run("00")

if __name__ == "__main__":
    main()