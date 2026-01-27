import argparse
import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenized_val_dir', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    p.add_argument('--bias_correction', action='store_true')
    return p.parse_args()


class TemperatureBiasScaler(torch.nn.Module):
    def __init__(self, num_classes, use_bias):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.zeros(1))
        self.use_bias = use_bias
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("bias", None)

    def forward(self, logits):
        temp = torch.exp(self.log_temp)
        out = logits / temp
        if self.use_bias:
            out = out + self.bias.unsqueeze(0)
        return out


def collect_logits(model, loader, device):
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.numpy())
    return np.vstack(logits_all), np.hstack(labels_all)


def fit_temperature_bias(logits, labels, use_bias, device):
    logits_t = torch.tensor(logits, dtype=torch.float32).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long).to(device)

    scaler = TemperatureBiasScaler(logits.shape[1], use_bias).to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    temp = float(torch.exp(scaler.log_temp).cpu())
    bias = scaler.bias.detach().cpu().numpy() if use_bias else None
    return temp, bias


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(args.checkpoint).to(args.device).eval()

    ds = load_from_disk(args.tokenized_val_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
        shuffle=False
    )

    logits, labels = collect_logits(model, loader, args.device)
    np.save(f"{args.output}/val_logits.npy", logits)
    np.save(f"{args.output}/val_labels.npy", labels)

    temp, bias = fit_temperature_bias(logits, labels, args.bias_correction, args.device)
    np.save(f"{args.output}/temperature.npy", np.array(temp))
    if bias is not None:
        np.save(f"{args.output}/bias.npy", bias)

    probs = softmax((logits / temp) + (bias if bias is not None else 0))
    np.save(f"{args.output}/val_probs_calibrated.npy", probs)

    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = precision_recall_fscore_support(labels, preds, average="macro")[2]

    print(f"Calibration done | temp={temp:.4f} | macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
