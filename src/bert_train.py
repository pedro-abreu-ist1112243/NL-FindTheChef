import torch
from transformers import BertTokenizer, BertModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import pandas as pd
from tqdm import tqdm
from logging import info
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
import os
from scipy.special import softmax


MODEL_NAME = "bert-base-uncased"


# Needed this hack to fix numpy int64 serialization issues
def _sanitize_for_json(obj):
    """
    Recursively sanitizes a dictionary or list by converting numpy types to
    native Python types. This version also handles numpy types in dictionary keys.
    """
    if isinstance(obj, dict):
        # Recursively sanitize both keys and values
        return {_sanitize_for_json(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class CustomTrainer(Trainer):
    """
    A custom trainer that performs a deep sanitization of the model config
    before saving to prevent JSON serialization errors with numpy types.
    """

    def _save(self, output_dir=None, state_dict=None):
        # Sanitize the entire config before saving
        config_dict = self.model.config.to_dict()
        sanitized_config_dict = _sanitize_for_json(config_dict)

        # Create a new config object from the sanitized dictionary
        # and assign it to the model
        new_config = self.model.config.from_dict(sanitized_config_dict)
        self.model.config = new_config

        # Now, call the original save method with the corrected config
        super()._save(output_dir, state_dict)


def get_cached_bert_model(output_dir: str = "./bert_model") -> CustomTrainer:
    """
    Loads the latest fine-tuned BERT model from a cached checkpoint directory.

    Args:
        output_dir (str): The directory where the model checkpoints are stored.
    Returns:
        CustomTrainer: A Trainer object configured with the loaded model and tokenizer, ready for inference or evaluation.
    Raises:
        FileNotFoundError: If the specified `output_dir` does not exist or if no valid checkpoints are found within it.
    """
    info(f"Attempting to load cached model from '{output_dir}'...")

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"The specified directory does not exist: {output_dir}")

    # Find all checkpoint subdirectories
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found in: {output_dir}")

    # Determine the latest checkpoint by the step number
    latest_checkpoint = max(checkpoints, key=lambda d: int(d.split("-")[1]))
    model_path = os.path.join(output_dir, latest_checkpoint)

    info(f"Loading model from the latest checkpoint: {model_path}")

    # Load the pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # A Trainer requires TrainingArguments, even for just inference
    training_args = TrainingArguments(output_dir=output_dir)

    # Instantiate the custom trainer with the loaded components
    trainer = CustomTrainer(model=model, args=training_args, tokenizer=tokenizer)

    info("Cached BERT model loaded successfully.")
    return trainer


def train_bert(
    df: pd.DataFrame,
    text_col: str = "text",
    random_state=42,
    val_size=0.2,
    output_dir: str = "./bert_model",
    force_use_cache: bool = False,
):
    """
    Fine-tunes a BERT model for text classification.

    Args:
        df (pd.DataFrame): DataFrame containing the text data and labels.
        text_col (str): Name of the column containing text data.
        random_state (int): Random seed for reproducibility.
        val_size (float): Proportion of the dataset to include in the validation split.
        output_dir (str): Directory to save the trained model and checkpoints.
    Returns:
        Trainer: The trained Hugging Face Trainer object.
    """
    info("Starting BERT training...")

    unique_chefs = df["chef_id"].unique()
    chef_to_id = {chef: int(i) for i, chef in enumerate(unique_chefs)}
    id_to_chef = {int(i): chef for i, chef in enumerate(unique_chefs)}
    NUM_LABELS = len(unique_chefs)

    # Convert to huggingface datasets
    df["label"] = df["chef_id"].map(chef_to_id)

    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=random_state, stratify=df["label"]
    )

    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Load the tokenizer associated with the chosen model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples[text_col], padding="max_length", truncation=True)

    info("Tokenizing dataset...")
    # Apply the tokenizer to the entire dataset
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    config.num_labels = NUM_LABELS
    config.id2label = id_to_chef
    config.label2id = chef_to_id

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,  # Pass the entire config object
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Computes accuracy for a batch of predictions."""
        predictions, labels = eval_pred
        # Get the index with the highest probability
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    latest_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda d: int(d.split("-")[1]))

    if latest_checkpoint:
        model_path = os.path.join(output_dir, latest_checkpoint)
        info(f"Found latest checkpoint in {model_path}. Loading from disk.")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Tokenize the validation set for evaluation
        tokenized_val_dataset = dataset_dict["validation"].map(
            lambda examples: tokenizer(
                examples[text_col], padding="max_length", truncation=True
            ),
            batched=True,
        )

        # Create a Trainer instance just for evaluation
        training_args = TrainingArguments(
            output_dir=output_dir, per_device_eval_batch_size=8
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        info("Evaluating cached model...")
        evaluation = trainer.evaluate()
        info(f"Cached model evaluation results: {evaluation}")
        return trainer

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Total number of training epochs
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=8,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True,  # Load the best model found during training
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    info("Training BERT model...")
    trainer.train()

    info("BERT training complete.")
    evaluation = trainer.evaluate()
    info(f"BERT evaluation results: {evaluation}")
    return trainer


def predict_with_bert(trainer: Trainer, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Uses a trained BERT model to make predictions on new text data.

    Args:
        trainer (Trainer): The trained Hugging Face Trainer object.
        df (pd.DataFrame): DataFrame containing the text data to predict on.
        text_col (str): Name of the column containing text data.
    Returns:
        pd.DataFrame: Array of predicted class labels.
    """
    info(f"Generating BERT predictions for {len(df)} samples...")
    # Ensure the trainer's tokenizer is used
    tokenizer = trainer.tokenizer
    if tokenizer is None:
        raise ValueError("Trainer must have a tokenizer to make predictions.")

    # Create a Dataset object and tokenize it
    predict_dataset = Dataset.from_pandas(df)
    tokenized_dataset = predict_dataset.map(
        lambda examples: tokenizer(
            examples[text_col], padding="max_length", truncation=True
        ),
        batched=True,
    )

    # Get model predictions (logits)
    predictions = trainer.predict(tokenized_dataset)
    logits = predictions.predictions

    # Convert logits to probabilities using softmax
    probabilities = softmax(logits, axis=1)

    # Create meaningful column names from the model's config
    id2label = trainer.model.config.id2label
    # Ensure keys are sorted to match softmax output order
    sorted_labels = [id2label[i] for i in sorted(id2label.keys())]
    bert_cols = [f"bert_prob_{label}" for label in sorted_labels]

    # Create a new DataFrame with the probabilities
    bert_features_df = pd.DataFrame(probabilities, columns=bert_cols, index=df.index)
    df_with_predictions = pd.concat([df, bert_features_df], axis=1)

    info("BERT prediction generation complete.")
    return df_with_predictions
