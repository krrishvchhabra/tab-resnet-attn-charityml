import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from src.preprocessing import load_and_preprocess_data
from src.model import TabResNetAttn, count_parameters
from src.train import train_one_epoch, evaluate


def measure_inference_latency(model, sample_num, sample_cat, device, num_runs=100, warmup=20):
    """
    Measures average inference latency per batch (in milliseconds).
    Includes warmup cycles to stabilize cache and clock frequencies.
    """
    model.eval()
    sample_num = sample_num.to(device)
    sample_cat = sample_cat.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_num, sample_cat)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(sample_num, sample_cat)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))  # milliseconds
            else:
                t0 = time.perf_counter()
                _ = model(sample_num, sample_cat)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)  # convert to ms

    return float(np.mean(timings))


def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Running Phase 2 Ablation Experiments on: {device}")

    train_loader, val_loader, test_loader, metadata = load_and_preprocess_data()
    criterion = nn.BCEWithLogitsLoss()

    # Grab a benchmark evaluation batch (256 samples)
    eval_batch_num, eval_batch_cat, _ = next(iter(test_loader))

    experiments = [
        {
            "name": "Baseline (Standard Linear Projection)",
            "use_embeddings": False,
            "use_residual": False,
            "use_attention": False
        },
        {
            "name": "Phase 1: Embeddings + Residual (Pre-Phase 2)",
            "use_embeddings": True,
            "use_residual": False,
            "use_attention": False
        },
        {
            "name": "Phase 1: Tab-ResNet-Attn (Standard Post-Act)",
            "use_embeddings": True,
            "use_residual": True,
            "use_attention": True
        }
    ]

    ablation_results = []

    for exp in experiments:
        print(f"\n[RUNNING] {exp['name']}...")
        model = TabResNetAttn(
            num_features_dim=metadata['num_features_dim'],
            cat_dims=metadata['cat_dims'],
            token_dim=32,
            num_heads=4,
            num_res_blocks=2,
            dropout=0.2,
            use_attention=exp['use_attention'],
            use_residual=exp['use_residual'],
            use_embeddings=exp['use_embeddings']
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 5

        best_val_f1 = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, _, v_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_state = model.state_dict()

        model.load_state_dict(best_state)
        _, test_acc, test_f05, test_auc, _, _ = evaluate(model, test_loader, criterion, device)
        batch_latency_ms = measure_inference_latency(model, eval_batch_num, eval_batch_cat, device)

        ablation_results.append({
            "Experiment": exp['name'],
            "Pre-Act & LayerScale": "✗",
            "SwiGLU & SE": "✗",
            "Self-Attention": "✓" if exp['use_attention'] else "✗",
            "Accuracy": round(test_acc, 4),
            "F0.5-Score": round(test_f05, 4),
            "ROC-AUC": round(test_auc, 4),
            "Parameters": f"{count_parameters(model):,}",
            "Batch Latency (ms)": round(batch_latency_ms, 2)
        })

    # Evaluate Phase 2 Best Model Checkpoint
    checkpoint = torch.load("models/proposed_best.pt", map_location=device)
    phase2_model = TabResNetAttn(
        num_features_dim=metadata['num_features_dim'],
        cat_dims=metadata['cat_dims'],
        token_dim=32,
        num_heads=4,
        num_res_blocks=2
    ).to(device)
    phase2_model.load_state_dict(checkpoint['model_state_dict'])
    _, p2_acc, p2_f05, p2_auc, _, _ = evaluate(phase2_model, test_loader, criterion, device)
    p2_batch_latency_ms = measure_inference_latency(phase2_model, eval_batch_num, eval_batch_cat, device)

    ablation_results.append({
        "Experiment": "Phase 2 Proposed (Pre-Act + SwiGLU + SE + LayerScale)",
        "Pre-Act & LayerScale": "✓",
        "SwiGLU & SE": "✓",
        "Self-Attention": "✓",
        "Accuracy": round(p2_acc, 4),
        "F0.5-Score": round(p2_f05, 4),
        "ROC-AUC": round(p2_auc, 4),
        "Parameters": f"{count_parameters(phase2_model):,}",
        "Batch Latency (ms)": round(p2_batch_latency_ms, 2)
    })

    df = pd.DataFrame(ablation_results)
    save_path = "results/ablation_study.csv"
    df.to_csv(save_path, index=False)
    print("\n" + "="*100)
    print("                      PHASE 2 ABLATION STUDY RESULTS WITH LATENCY                      ")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    print(f"[INFO] Updated ablation table saved to {save_path}")


if __name__ == '__main__':
    run_ablation()