# train_codebert_v2_focal.py
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



# Trainer  FOCAL LOSS  (without class weights)

class FocalLossTrainerV2F(Trainer):
    def __init__(self, *args, focal_gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()

        return (loss, outputs) if return_outputs else loss




class CodeBERTFocalTrainerV2F:
    def __init__(self, model_name="microsoft/codebert-base", num_labels=11, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def load_data(self, train_path, val_path):
        train_df = pd.read_parquet(train_path, columns=["code", "label"])
        val_df   = pd.read_parquet(val_path, columns=["code", "label"])
        return train_df, val_df

    def prepare_datasets(self, train_df, val_df, tokenized_dir, force_retokenize=False):
        if os.path.exists(tokenized_dir) and not force_retokenize:
            self.train_dataset = load_from_disk(os.path.join(tokenized_dir, "train"))
            self.val_dataset   = load_from_disk(os.path.join(tokenized_dir, "val"))
            return

        raw_train = Dataset.from_pandas(train_df[["code", "label"]])
        raw_val   = Dataset.from_pandas(val_df[["code", "label"]])

        def tokenize(batch):
            return self.tokenizer(
                batch["code"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        self.train_dataset = raw_train.map(tokenize, batched=True)
        self.val_dataset   = raw_val.map(tokenize, batched=True)

        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        self.val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        self.train_dataset.save_to_disk(os.path.join(tokenized_dir, "train"))
        self.val_dataset.save_to_disk(os.path.join(tokenized_dir, "val"))

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds  = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    def train(
        self,
        output_dir,
        epochs,
        batch_size,
        lr,
        grad_accum,
        fp16=True
    ):
        model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=2000,
            save_steps=2000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=fp16,
            save_total_limit=2,
            max_grad_norm=1.0,
            report_to="none"
        )

        trainer = FocalLossTrainerV2F(
            model=model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            focal_gamma=1.0
        )

        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))




def run_experiment(
    train_path,
    val_path,
    output_dir,
    tokenized_dir,
    epochs=2,
    batch_size=2,
    grad_accum=16,
    lr=2e-5,
    force_retokenize=False
):
    os.makedirs(output_dir, exist_ok=True)

    trainer = CodeBERTFocalTrainerV2F(max_length=512)
    train_df, val_df = trainer.load_data(train_path, val_path)
    trainer.prepare_datasets(train_df, val_df, tokenized_dir, force_retokenize)

    trainer.train(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_accum=grad_accum,
        fp16=torch.cuda.is_available()
    )
