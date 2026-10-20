import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, fbeta_score, roc_auc_score

from src.preprocessing import load_and_preprocess_data
from src.model import TabularMLP, TabResNetAttn, count_parameters


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x_num, x_cat, y in dataloader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    epoch_acc = accuracy_score(all_targets, (all_preds >= 0.5).astype(int))
    epoch_f1 = fbeta_score(all_targets, (all_preds >= 0.5).astype(int), beta=0.5, zero_division=0)
    epoch_auc = roc_auc_score(all_targets, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, epoch_auc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x_num, x_cat, y in dataloader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        logits = model(x_num, x_cat)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    epoch_acc = accuracy_score(all_targets, (all_preds >= 0.5).astype(int))
    epoch_f1 = fbeta_score(all_targets, (all_preds >= 0.5).astype(int), beta=0.5, zero_division=0)
    epoch_auc = roc_auc_score(all_targets, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, epoch_auc, all_preds, all_targets


def plot_learning_curves(history, save_path="results/training_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curve
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Loss Convergence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    axes[1].plot(epochs, history['train_acc'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Accuracy Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # F-beta (beta=0.5) Curve
    axes[2].plot(epochs, history['train_f1'], label='Train F0.5', color='#1f77b4', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], label='Val F0.5', color='#ff7f0e', linewidth=2)
    axes[2].set_title('F-beta (β=0.5) Score Curve', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F-Score')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Learning curves saved to {save_path}")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using compute device: {device}")

    # 1. Load data
    train_loader, val_loader, test_loader, metadata = load_and_preprocess_data(
        data_path=args.data_path,
        batch_size=args.batch_size
    )

    # 2. Instantiate Model
    if args.model_type == 'mlp':
        model = TabularMLP(
            num_features_dim=metadata['num_features_dim'],
            cat_dims=metadata['cat_dims'],
            hidden_dims=[128, 64],
            dropout=args.dropout
        )
    elif args.model_type == 'proposed':
        model = TabResNetAttn(
            num_features_dim=metadata['num_features_dim'],
            cat_dims=metadata['cat_dims'],
            token_dim=32,
            num_heads=4,
            num_res_blocks=2,
            dropout=args.dropout,
            use_attention=True,
            use_residual=True,
            use_embeddings=True
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model.to(device)
    total_params = count_parameters(model)
    print(f"[INFO] Model Architecture: {args.model_type.upper()} | Trainable Parameters: {total_params:,}")

    # 3. Objective, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. Training Loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }

    best_val_f1 = -1.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_best.pt")

    print("\n" + "="*75)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'Val Acc':<12}{'Val F0.5':<12}{'Val AUC':<12}")
    print("="*75)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, v_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(v_f1)
        history['train_auc'].append(tr_auc)
        history['val_auc'].append(v_auc)

        print(f"{epoch:<8}{tr_loss:<12.4f}{v_loss:<12.4f}{v_acc:<12.4f}{v_f1:<12.4f}{v_auc:<12.4f}")

        # Checkpoint based on Validation F-beta score
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': v_f1,
                'metadata': metadata,
                'args': vars(args)
            }, best_model_path)

    print("="*75)
    print(f"[INFO] Best Validation F-score: {best_val_f1:.4f} saved to {best_model_path}")

    # 5. Export training curve plot
    curve_path = os.path.join(args.results_dir, f"{args.model_type}_training_curve.png")
    plot_learning_curves(history, save_path=curve_path)

    # 6. Save model description summary (required in Section 6)
    desc_path = os.path.join(args.checkpoint_dir, "model_description.txt")
    with open(desc_path, "w") as f:
        f.write(f"Project: Finding Donors for CharityML using Supervised Deep Learning\n")
        f.write(f"Model Type: {args.model_type.upper()}\n")
        f.write(f"Total Trainable Parameters: {total_params:,}\n")
        f.write(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})\n")
        f.write(f"Scheduler: CosineAnnealingLR\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Best Val F0.5 Score: {best_val_f1:.4f}\n\n")
        f.write(str(model))

    return best_model_path, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Tabular Deep Learning Models for CharityML")
    parser.add_argument('--data_path', type=str, default='data/census.csv')
    parser.add_argument('--model_type', type=str, default='proposed', choices=['mlp', 'proposed'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--checkpoint_dir', type=str, default='models')
    parser.add_argument('--results_dir', type=str, default='results')

    args = parser.parse_args()
    train(args)
