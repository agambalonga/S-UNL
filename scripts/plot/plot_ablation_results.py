#!/usr/bin/env python3
"""
Script per generare grafici dall'ablation study su parafrasi/epoche e metodi di unlearning.

Genera grafici che mostrano come variano le metriche TOFU aggregate:
1. Al variare del numero di parafrasi (5, 10, 15, 20) - epoca finale
2. Al variare delle epoche (5, 10, 15, 20) - per ogni configurazione di parafrasi

Usage:
    python scripts/plot_ablation_results.py
    python scripts/plot_ablation_results.py --paraphrases 5 10 15 20
    python scripts/plot_ablation_results.py --epochs 5 10 15 20
"""

import json
import os
import glob
import argparse
import re
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configurazione stile grafici
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Metriche da plottare
METRICS_TO_PLOT = [
    'memorization_score',
    'privacy_score', 
    'utility_score',
    'aggregate_score'
]

METRIC_LABELS = {
    'memorization_score': 'Memorization Score (↑ better)',
    'privacy_score': 'Privacy Score (↑ better)',
    'utility_score': 'Utility Score (↑ better)',
    'aggregate_score': 'Aggregate Score (↑ better)'
}

TRAINERS = ['GradDiff', 'NPO', 'SimNPO', 'DPO', 'RMU', 'UNDIAL']


def extract_checkpoint_number(path: str) -> int:
    """Estrae il numero di step dal checkpoint path."""
    match = re.search(r'checkpoint-(\d+)', path)
    return int(match.group(1)) if match else 0


def load_results_by_paraphrases(base_dir: str = "saves/unlearn", 
                                paraphrase_counts: List[int] = [0, 5, 10, 15, 20]) -> pd.DataFrame:
    """
    Carica risultati finali variando le parafrasi.
    Prende l'ultimo checkpoint (epoca 20) per ogni configurazione.
    
    Returns:
        DataFrame con colonne: trainer, num_paraphrases, metric, value
    """
    results = []
    
    for trainer in TRAINERS:
        for num_para in paraphrase_counts:
            # Pattern base experiment directory
            pattern = f"{base_dir}/tofu_Llama-3.2-1B-Instruct_forget10_{trainer}_para{num_para}_ep20"
            exp_dirs = glob.glob(pattern)
            
            if not exp_dirs:
                print(f"⚠️  Missing: {trainer} with {num_para} paraphrases")
                continue
            
            exp_dir = exp_dirs[0]
            
            # Trova tutti i checkpoint con eval
            checkpoint_dirs = []
            for ckpt_dir in glob.glob(f"{exp_dir}/checkpoint-*"):
                eval_file = os.path.join(ckpt_dir, 'evals', 'TOFU_SUMMARY.json')
                if os.path.exists(eval_file):
                    step = extract_checkpoint_number(ckpt_dir)
                    checkpoint_dirs.append((step, ckpt_dir, eval_file))
            
            if not checkpoint_dirs:
                print(f"⚠️  No evaluated checkpoints: {trainer} para={num_para}")
                continue
            
            # Ordina per step e prendi l'ultimo (epoca 20)
            checkpoint_dirs.sort(key=lambda x: x[0])
            best_step, best_ckpt, summary_file = checkpoint_dirs[-1]
            
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                for metric in METRICS_TO_PLOT:
                    if metric in data:
                        results.append({
                            'trainer': trainer,
                            'num_paraphrases': num_para,
                            'metric': metric,
                            'value': data[metric]
                        })
                
                print(f"✅ Loaded: {trainer} para={num_para} (step={best_step}, last checkpoint)")
                
            except Exception as e:
                print(f"❌ Error loading {summary_file}: {e}")
    
    return pd.DataFrame(results)


def load_results_by_epochs(base_dir: str = "saves/unlearn",
                           paraphrase_counts: List[int] = [0, 5, 10, 15, 20],
                           epoch_checkpoints: List[int] = [5, 10, 15, 20]) -> pd.DataFrame:
    """
    Carica risultati a epoche specifiche per ogni configurazione di parafrasi.
    L'N-esimo checkpoint ordinato per step corrisponde all'epoca N.
    
    Returns:
        DataFrame con colonne: trainer, num_paraphrases, epoch, metric, value
    """
    results = []
    
    for trainer in TRAINERS:
        for num_para in paraphrase_counts:
            # Pattern base
            pattern_base = f"{base_dir}/tofu_Llama-3.2-1B-Instruct_forget10_{trainer}_para{num_para}_ep*"
            exp_dirs = glob.glob(pattern_base)
            
            if not exp_dirs:
                continue
            
            exp_dir = exp_dirs[0]  # Dovrebbe essere uno solo
            
            # Trova tutti i checkpoint con eval, ordinati per step
            checkpoint_list = []
            for ckpt_dir in glob.glob(f"{exp_dir}/checkpoint-*"):
                eval_file = os.path.join(ckpt_dir, 'evals', 'TOFU_SUMMARY.json')
                if os.path.exists(eval_file):
                    step = extract_checkpoint_number(ckpt_dir)
                    checkpoint_list.append((step, eval_file))
            
            if not checkpoint_list:
                print(f"⚠️  No evaluated checkpoints: {trainer} para={num_para}")
                continue
            
            # Ordina per step: 1° checkpoint = epoca 1, 2° = epoca 2, ecc.
            checkpoint_list.sort(key=lambda x: x[0])
            
            # Carica dati per le epoche richieste
            for target_epoch in epoch_checkpoints:
                # Indice 0-based: epoca 1 = index 0, epoca 5 = index 4
                checkpoint_index = target_epoch - 1
                
                if checkpoint_index >= len(checkpoint_list):
                    print(f"⚠️  Missing: {trainer} para={num_para} epoch={target_epoch} "
                          f"(only {len(checkpoint_list)} checkpoints available)")
                    continue
                
                step, summary_file = checkpoint_list[checkpoint_index]
                
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                    
                    for metric in METRICS_TO_PLOT:
                        if metric in data:
                            results.append({
                                'trainer': trainer,
                                'num_paraphrases': num_para,
                                'epoch': target_epoch,
                                'metric': metric,
                                'value': data[metric]
                            })
                    
                    print(f"✅ Loaded: {trainer} para={num_para} epoch={target_epoch} (step={step}, checkpoint #{target_epoch})")
                    
                except Exception as e:
                    print(f"❌ Error loading {summary_file}: {e}")
    
    return pd.DataFrame(results)


def plot_metric_by_paraphrases(df: pd.DataFrame, metric: str, output_dir: str):
    """Plotta metrica vs numero parafrasi (epoca finale)."""
    os.makedirs(output_dir, exist_ok=True)
    
    metric_df = df[df['metric'] == metric]
    
    if metric_df.empty:
        print(f"⚠️  No data for metric: {metric}")
        return
    
    plt.figure(figsize=(10, 6))
    
    for trainer in TRAINERS:
        trainer_df = metric_df[metric_df['trainer'] == trainer]
        if not trainer_df.empty:
            plt.plot(
                trainer_df['num_paraphrases'],
                trainer_df['value'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=trainer
            )
    
    plt.xlabel('Number of Paraphrases', fontsize=13)
    plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    plt.title(f'{METRIC_LABELS.get(metric, metric)} vs Number of Paraphrases (Final Epoch)', 
              fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f"{output_dir}/{metric}_vs_paraphrases.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def plot_metric_by_epochs(df: pd.DataFrame, metric: str, num_paraphrases: int, output_dir: str):
    """Plotta metrica vs epoche per una configurazione di parafrasi."""
    os.makedirs(output_dir, exist_ok=True)
    
    metric_df = df[(df['metric'] == metric) & (df['num_paraphrases'] == num_paraphrases)]
    
    if metric_df.empty:
        print(f"⚠️  No data for metric: {metric}, paraphrases: {num_paraphrases}")
        return
    
    plt.figure(figsize=(10, 6))
    
    for trainer in TRAINERS:
        trainer_df = metric_df[metric_df['trainer'] == trainer]
        if not trainer_df.empty:
            # Ordina per epoca
            trainer_df = trainer_df.sort_values('epoch')
            plt.plot(
                trainer_df['epoch'],
                trainer_df['value'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=trainer
            )
    
    plt.xlabel('Training Epochs', fontsize=13)
    plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    plt.title(f'{METRIC_LABELS.get(metric, metric)} vs Epochs ({num_paraphrases} Paraphrases)', 
              fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f"{output_dir}/{metric}_vs_epochs_para{num_paraphrases}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def plot_all_paraphrases_grid(df: pd.DataFrame, output_dir: str):
    """Grid 2x2 con tutte le metriche vs parafrasi."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]
        metric_df = df[df['metric'] == metric]
        
        if metric_df.empty:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        for trainer in TRAINERS:
            trainer_df = metric_df[metric_df['trainer'] == trainer]
            if not trainer_df.empty:
                ax.plot(
                    trainer_df['num_paraphrases'],
                    trainer_df['value'],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=trainer
                )
        
        ax.set_xlabel('Number of Paraphrases', fontsize=12)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('TOFU Metrics vs Number of Paraphrases (Final Epoch)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = f"{output_dir}/all_metrics_vs_paraphrases_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def plot_all_epochs_grid(df: pd.DataFrame, num_paraphrases: int, output_dir: str):
    """Grid 2x2 con tutte le metriche vs epoche."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]
        metric_df = df[(df['metric'] == metric) & (df['num_paraphrases'] == num_paraphrases)]
        
        if metric_df.empty:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        for trainer in TRAINERS:
            trainer_df = metric_df[metric_df['trainer'] == trainer]
            if not trainer_df.empty:
                trainer_df = trainer_df.sort_values('epoch')
                ax.plot(
                    trainer_df['epoch'],
                    trainer_df['value'],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=trainer
                )
        
        ax.set_xlabel('Training Epochs', fontsize=12)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'TOFU Metrics vs Epochs ({num_paraphrases} Paraphrases)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = f"{output_dir}/all_metrics_vs_epochs_para{num_paraphrases}_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def generate_summary_tables(df_para: pd.DataFrame, df_epochs: pd.DataFrame, output_dir: str):
    """Genera tabelle CSV riassuntive."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Tabella parafrasi
    if not df_para.empty:
        summary_para = df_para.pivot_table(
            index=['trainer', 'num_paraphrases'],
            columns='metric',
            values='value'
        ).reset_index()
        
        output_file = f"{output_dir}/results_by_paraphrases.csv"
        summary_para.to_csv(output_file, index=False, float_format='%.4f')
        print(f"📋 Saved: {output_file}")
        
        print("\n" + "="*100)
        print("RESULTS BY PARAPHRASES (Final Epoch)")
        print("="*100)
        print(summary_para.to_string(index=False))
        print("="*100 + "\n")
    
    # Tabella epoche (per ogni config di parafrasi)
    if not df_epochs.empty:
        for num_para in df_epochs['num_paraphrases'].unique():
            summary_epochs = df_epochs[df_epochs['num_paraphrases'] == num_para].pivot_table(
                index=['trainer', 'epoch'],
                columns='metric',
                values='value'
            ).reset_index()
            
            output_file = f"{output_dir}/results_by_epochs_para{num_para}.csv"
            summary_epochs.to_csv(output_file, index=False, float_format='%.4f')
            print(f"📋 Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot ablation study results')
    parser.add_argument('--base-dir', default='saves/unlearn', help='Base directory for results')
    parser.add_argument('--output-dir', default='plots/ablation_study', help='Output directory for plots')
    parser.add_argument('--paraphrases', nargs='+', type=int, default=[0, 5, 10, 15, 20],
                       help='Paraphrase counts to analyze (0=baseline without paraphrases)')
    parser.add_argument('--epochs', nargs='+', type=int, default=[5, 10, 15, 20],
                       help='Epoch checkpoints to analyze')
    
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("TOFU Ablation Study - Results Plotting")
    print("="*100 + "\n")
    
    # Carica risultati per parafrasi (epoca finale)
    print("📂 Loading results by paraphrases (final epoch)...")
    df_para = load_results_by_paraphrases(args.base_dir, args.paraphrases)
    
    # Carica risultati per epoche
    print("\n📂 Loading results by epochs...")
    df_epochs = load_results_by_epochs(args.base_dir, args.paraphrases, args.epochs)
    
    if df_para.empty and df_epochs.empty:
        print("\n❌ No results found! Make sure to run the training first.")
        return
    
    print("\n✅ Loaded data:")
    if not df_para.empty:
        print(f"   - Paraphrases: {len(df_para)} data points")
    if not df_epochs.empty:
        print(f"   - Epochs: {len(df_epochs)} data points")
    
    # Genera grafici vs parafrasi
    if not df_para.empty:
        print("\n📊 Generating plots vs paraphrases...\n")
        for metric in METRICS_TO_PLOT:
            plot_metric_by_paraphrases(df_para, metric, args.output_dir)
        plot_all_paraphrases_grid(df_para, args.output_dir)
    
    # Genera grafici vs epoche (per ogni config di parafrasi)
    if not df_epochs.empty:
        print("\n📊 Generating plots vs epochs...\n")
        for num_para in args.paraphrases:
            for metric in METRICS_TO_PLOT:
                plot_metric_by_epochs(df_epochs, metric, num_para, args.output_dir)
            plot_all_epochs_grid(df_epochs, num_para, args.output_dir)
    
    # Genera tabelle
    print("\n📋 Generating summary tables...\n")
    generate_summary_tables(df_para, df_epochs, args.output_dir)
    
    print("\n" + "="*100)
    print(f"✅ All plots saved in: {args.output_dir}/")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

