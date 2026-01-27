# train_codebert_v3.py
import os
import warnings
warnings.filterwarnings("ignore")

import re
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight


# preprocessing

def preprocess_code(code: str) -> str:
    if not isinstance(code, str):
        return ""

    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'#.*', '', code)

    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.S)

    # Remove Python docstrings
    code = re.sub(r'""".*?"""', '', code, flags=re.S)
    code = re.sub(r"'''.*?'''", '', code, flags=re.S)

    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)

    return code.strip()




class WeightedLossTrainerv3(Trainer):
    def __init__(self, *args, weight=None, use_focal=False, focal_gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.weight is not None:
            self.weight = self.weight.to(logits.device)

        if self.use_focal:
            ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        else:
            loss = F.cross_entropy(logits, labels, weight=self.weight)

        return (loss, outputs) if return_outputs else loss


class CodeBERTTrainerV3:
    def __init__(self, model_name='microsoft/codebert-base', num_labels=11, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.class_weights = None

    def load_and_prepare_data(self, train_path, val_path):
        print("Loading datasets...")

        train_df = pd.read_parquet(train_path, columns=['code', 'label'])
        val_df = pd.read_parquet(val_path, columns=['code', 'label'])

        print("Applying preprocessing...")
        train_df['code'] = train_df['code'].apply(preprocess_code)
        val_df['code'] = val_df['code'].apply(preprocess_code)

        y_train = train_df['label'].values
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        return train_df, val_df

    def prepare_datasets(self, train_df, val_df, tokenized_dir, force_retokenize=False):
        if os.path.exists(tokenized_dir) and not force_retokenize:
            self.train_dataset = load_from_disk(os.path.join(tokenized_dir, 'train'))
            self.val_dataset = load_from_disk(os.path.join(tokenized_dir, 'val'))
            return

        raw_train = Dataset.from_pandas(train_df[['code', 'label']])
        raw_val = Dataset.from_pandas(val_df[['code', 'label']])

        def tokenize_function(examples):
            return self.tokenizer(
                examples['code'],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        self.train_dataset = raw_train.map(tokenize_function, batched=True)
        self.val_dataset = raw_val.map(tokenize_function, batched=True)

        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        self.train_dataset.save_to_disk(os.path.join(tokenized_dir, 'train'))
        self.val_dataset.save_to_disk(os.path.join(tokenized_dir, 'val'))

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        acc = accuracy_score(labels, preds)

        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def run_training(self, output_dir, num_epochs, batch_size, learning_rate,
                     grad_accum=1, use_focal=False, fp16=True):

        model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        if use_focal==False:
            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size * 2,
                gradient_accumulation_steps=grad_accum,
                learning_rate=learning_rate,
                weight_decay=0.01,
                evaluation_strategy="steps",
                eval_steps=1000,
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                fp16=fp16,
                save_total_limit=2,
                report_to="none"
            )

            trainer = WeightedLossTrainer(
                model=model,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                compute_metrics=self.compute_metrics,
                weight=self.class_weights,
                use_focal=use_focal
            )
        else: 
            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size * 2,
                gradient_accumulation_steps=grad_accum,
                learning_rate=learning_rate,
                weight_decay=0.01,
                evaluation_strategy="steps",
                eval_steps=1000,
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                fp16=fp16,
                max_grad_norm=1.0,
                save_total_limit=2,
                report_to="none"
                
            )

            trainer = WeightedLossTrainer(
                model=model,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                compute_metrics=self.compute_metrics,
                weight=None,
                use_focal=use_focal
            )

        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
