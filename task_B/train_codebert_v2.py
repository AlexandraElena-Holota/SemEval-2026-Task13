import os
import argparse
import warnings
import shutil
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

#trainer weighted loss (cross entropy or focal loss)
class WeightedLossTrainerV2(Trainer):
    def __init__(self, *args, weight=None, use_focal=False, focal_gamma=2.0, **kwargs):
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
            # Focal Loss implementation
            ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
            loss = focal_loss
        else:
            # Standard Weighted Cross Entropy 
            loss = F.cross_entropy(logits, labels, weight=self.weight)

        return (loss, outputs) if return_outputs else loss



class CodeBERTTrainerV2:
    def __init__(self, model_name='microsoft/codebert-base', num_labels=11, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length  
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = None
        
        self.label2id = None 
        self.id2label = None
        
        self.class_weights = None
        self.train_dataset = None
        self.val_dataset = None

    def load_and_prepare_data(self, train_path, val_path):
        print(f"Loading data from {train_path} and {val_path}...")
        if train_path.endswith('.parquet'):
            train_df = pd.read_parquet(train_path, columns=['code', 'label'])
            val_df = pd.read_parquet(val_path, columns=['code', 'label'])
        else:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)

        print("Computing class weights...")
        y_train = train_df['label'].values
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class Weights calcolati: {self.class_weights}")

        return train_df, val_df

    def prepare_datasets(self, train_df, val_df, tokenized_dir, force_retokenize=False):
        if os.path.exists(tokenized_dir) and not force_retokenize:
            print(f"Loading tokenized datasets from {tokenized_dir}...")
            try:
                self.train_dataset = load_from_disk(os.path.join(tokenized_dir, 'train'))
                self.val_dataset = load_from_disk(os.path.join(tokenized_dir, 'val'))
                return
            except Exception as e:
                print(f"Failed to load. Retokenizing... Error: {e}")

        print("Tokenizing data (single process to save RAM)...")

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

        print(f"Saving tokenized datasets to {tokenized_dir}...")
        self.train_dataset.save_to_disk(os.path.join(tokenized_dir, 'train'))
        self.val_dataset.save_to_disk(os.path.join(tokenized_dir, 'val'))

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def run_training(self, output_dir, num_epochs, batch_size, learning_rate, 
                     grad_accum=1, use_sampler=False, use_focal=False, fp16=True, 
                     resume_from_checkpoint=None):
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2, 
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=1000, 
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=fp16,
            dataloader_num_workers=2,
            save_total_limit=2, 
            report_to="none"
        )

        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            weight=self.class_weights, 
            use_focal=use_focal
        )

        # Sampler Custom 
        if use_sampler:
            
            train_labels = [self.train_dataset[i]['label'].item() for i in range(len(self.train_dataset))]
            sample_weights = [self.class_weights[l] for l in train_labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            def get_train_dataloader_patched():
                return DataLoader(
                    self.train_dataset, 
                    batch_size=batch_size, 
                    sampler=sampler,
                    collate_fn=DataCollatorWithPadding(self.tokenizer),
                    num_workers=2
                )
            trainer.get_train_dataloader = get_train_dataloader_patched

        print("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        print("Saving final model...")
        trainer.save_model(os.path.join(output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

    
    def run_experiment(train_path, val_path, output_dir, tokenized_dir, 
                   epochs=2, batch_size=4, grad_accum=8, lr=2e-5, 
                   force_retokenize=False):
    
        os.makedirs(output_dir, exist_ok=True)
    
        trainer_wrapper = CodeBERTTrainer(max_length=512) 
    
        
        train_df, val_df = trainer_wrapper.load_and_prepare_data(train_path, val_path)
    
        trainer_wrapper.prepare_datasets(train_df, val_df, tokenized_dir, force_retokenize=force_retokenize)
    
        
        
        trainer_wrapper.run_training(
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            grad_accum=grad_accum,
            use_sampler=False, 
            use_focal=False,
            fp16=torch.cuda.is_available()
        )

if __name__ == "__main__":
    pass