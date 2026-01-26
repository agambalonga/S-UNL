#!/usr/bin/env python3
"""
Script per analizzare l'evoluzione delle metriche attraverso le epoche
Genera grafici che mostrano le dinamiche temporali dell'unlearning
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import json
import glob
import os
import re

# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Metriche da analizzare con direzioni ottimali
METRICS_CONFIG = {
    'exact_memorization': {
        'label': 'Exact Memorization',
        'direction': 'lower',  # più basso è meglio
        'color': 'Blues'
    },
    'extraction_strength': {
        'label': 'Extraction Strength',
        'direction': 'lower',
        'color': 'Oranges'
    },
    'forget_Q_A_Prob': {
        'label': 'Forget Q+A Probability',
        'direction': 'lower',
        'color': 'Greens'
    },
    'forget_Q_A_PARA_Prob': {
        'label': 'Forget Paraphrased Probability',
        'direction': 'lower',
        'color': 'Purples'
    },
    'forget_truth_ratio': {
        'label': 'Truth Ratio',
        'direction': 'lower',  # più vicino a 0.5 è meglio
        'color': 'Reds'
    },
    'mia_loss': {
        'label': 'MIA Loss',
        'direction': 'higher',  # più alto è meglio
        'color': 'YlOrBr'
    },
    'model_utility': {
        'label': 'Model Utility',
        'direction': 'higher',
        'color': 'GnBu'
    },
    'forget_Q_A_gibberish': {
        'label': 'Forget Fluency',
        'direction': 'higher',  # più alto è meglio (più fluente)
        'color': 'RdPu'
    },
    'privleak': {
        'label': 'Privacy Leakage',
        'direction': 'zero',  # più vicino a 0 è meglio
        'color': 'coolwarm'
    }
}

METHODS = ['DPO', 'GradDiff', 'NPO', 'RMU', 'SimNPO', 'UNDIAL']
PARAPHRASE_CONFIGS = [0, 5, 10, 15, 20]
EPOCHS = [5, 10, 15, 20]

# Metriche aggregate da escludere
AGGREGATE_METRICS = [
    'memorization_score',
    'privacy_score',
    'utility_score',
    'aggregate_score'
]


def extract_checkpoint_number(path: str) -> int:
    """Estrae il numero di step dal checkpoint path."""
    match = re.search(r'checkpoint-(\d+)', path)
    return int(match.group(1)) if match else 0


def load_data(base_dir='saves/unlearn', target_epochs=None):
    """
    Carica i dati direttamente dai checkpoint, usando la stessa logica di plot_detailed_metrics.py
    
    Args:
        base_dir: Directory base dei risultati
        target_epochs: Liste di epoche da analizzare
        
    Returns:
        DataFrame con colonne: trainer, num_paraphrases, epoch, metric_name, metric_value
    """
    if target_epochs is None:
        target_epochs = EPOCHS
    
    results = []
    
    for method in METHODS:
        for num_para in PARAPHRASE_CONFIGS:
            # Pattern experiment directory
            pattern = f"{base_dir}/tofu_Llama-3.2-1B-Instruct_forget10_{method}_para{num_para}_ep20"
            exp_dirs = glob.glob(pattern)
            
            if not exp_dirs:
                print(f"⚠️  Missing: {method} with {num_para} paraphrases")
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
                print(f"⚠️  No evaluated checkpoints: {method} para={num_para}")
                continue
            
            # Ordina per step
            checkpoint_dirs.sort(key=lambda x: x[0])
            max_step = checkpoint_dirs[-1][0]
            
            # Calcola step per epoca (assumendo che max_step corrisponda a 20 epoche)
            steps_per_epoch = max_step / 20.0
            
            # Per ogni epoca target, trova il checkpoint più vicino
            for target_epoch in target_epochs:
                target_step = int(target_epoch * steps_per_epoch)
                
                # Trova checkpoint più vicino a target_step
                closest_ckpt = min(checkpoint_dirs, key=lambda x: abs(x[0] - target_step))
                actual_step, ckpt_dir, summary_file = closest_ckpt
                actual_epoch = round(actual_step / steps_per_epoch)
                
                # Salta se l'epoca effettiva non corrisponde (tolleranza ±0.5 epoche)
                if abs(actual_epoch - target_epoch) > 0.5:
                    print(f"⚠️  No checkpoint near epoch {target_epoch}: {method} para={num_para}")
                    continue
                
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                    
                    # Estrai TUTTE le metriche (escluse quelle aggregate)
                    for metric, value in data.items():
                        if metric not in AGGREGATE_METRICS:
                            results.append({
                                'trainer': method,
                                'num_paraphrases': num_para,
                                'epoch': target_epoch,
                                metric: value
                            })
                    
                    # Stampa solo il primo caricamento per metodo/para
                    if target_epoch == target_epochs[0]:
                        print(f"✅ {method} para={num_para}: found {len(checkpoint_dirs)} checkpoints")
                    
                except Exception as e:
                    print(f"❌ Error loading {summary_file}: {e}")
    
    if not results:
        print("\n❌ No data loaded! Check that checkpoints have been evaluated.")
        return pd.DataFrame()
    
    # Converti lista di dict in DataFrame
    # Ogni riga ha trainer, num_paraphrases, epoch + tutte le metriche come colonne
    df = pd.DataFrame(results)
    
    # Raggruppa per (trainer, num_paraphrases, epoch) e prendi la prima riga
    # (potrebbero esserci duplicati se abbiamo processato lo stesso checkpoint)
    df = df.groupby(['trainer', 'num_paraphrases', 'epoch'], as_index=False).first()
    
    print(f"\n✓ Loaded {len(df)} rows from checkpoints")
    print(f"  Methods: {sorted(df['trainer'].unique())}")
    print(f"  Epochs: {sorted(df['epoch'].unique())}")
    print(f"  Paraphrases: {sorted(df['num_paraphrases'].unique())}")
    
    return df


def plot_method_evolution(df, method, metric, output_dir):
    """
    Grafico evoluzione di una metrica attraverso le epoche per un metodo
    Linee separate per ogni configurazione di parafrasi
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    method_data = df[df['trainer'] == method]
    config = METRICS_CONFIG[metric]
    
    # Linea per ogni configurazione di parafrasi
    for para_count in PARAPHRASE_CONFIGS:
        para_data = method_data[method_data['num_paraphrases'] == para_count]
        para_data = para_data.sort_values('epoch')
        
        ax.plot(para_data['epoch'], para_data[metric], 
                marker='o', linewidth=2, markersize=8,
                label=f'para{para_count}')
    
    ax.set_xlabel('Epoca', fontsize=18, fontweight='bold')
    ax.set_ylabel(config['label'], fontsize=18, fontweight='bold')
    ax.set_title(f'{method}: Evoluzione {config["label"]} attraverso le Epoche',
                 fontsize=18, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=18)  # assi principali
    ax.legend(title='Configurazione', fontsize=18, title_fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(EPOCHS)
    
    # Aggiungi indicazione direzione ottimale
    direction_text = {
        'lower': '↓ Migliore',
        'higher': '↑ Migliore',
        'zero': '→ 0 Migliore'
    }
    ax.text(0.02, 0.98, direction_text[config['direction']], 
            transform=ax.transAxes, fontsize=16, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f'{method}_{metric}_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_all_methods_comparison(df, metric, epoch, output_dir):
    """
    Grafico a barre raggruppate: confronto tra tutti i metodi a una specifica epoca
    Barre separate per ogni configurazione di parafrasi
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    epoch_data = df[df['epoch'] == epoch]
    config = METRICS_CONFIG[metric]
    
    # Prepara dati per barre raggruppate
    x = np.arange(len(METHODS))
    width = 0.15
    
    for i, para_count in enumerate(PARAPHRASE_CONFIGS):
        para_data = epoch_data[epoch_data['num_paraphrases'] == para_count]
        values = [para_data[para_data['trainer'] == m][metric].values[0] 
                  if len(para_data[para_data['trainer'] == m]) > 0 else 0
                  for m in METHODS]
        
        offset = width * (i - 2)  # Centra le barre
        ax.bar(x + offset, values, width, label=f'para{para_count}', alpha=0.8)
    
    ax.set_xlabel('Metodo', fontsize=12, fontweight='bold')
    ax.set_ylabel(config['label'], fontsize=12, fontweight='bold')
    ax.set_title(f'Confronto {config["label"]} tra Metodi (Epoca {epoch})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontweight='bold')
    ax.legend(title='Configurazione', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Indicazione direzione ottimale
    direction_text = {
        'lower': '↓ Migliore',
        'higher': '↑ Migliore',
        'zero': '→ 0 Migliore'
    }
    ax.text(0.02, 0.98, direction_text[config['direction']], 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f'comparison_{metric}_epoch{epoch}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_convergence_heatmap(df, method, metric, output_dir):
    """
    Heatmap: righe = epoche, colonne = num_paraphrases
    Mostra come il metodo converge nel tempo con diverse parafrasi
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    method_data = df[df['trainer'] == method]
    config = METRICS_CONFIG[metric]
    
    # Pivot per creare matrice epoca x parafrasi
    pivot_data = method_data.pivot(index='epoch', columns='num_paraphrases', values=metric)
    pivot_data = pivot_data.reindex(index=EPOCHS, columns=PARAPHRASE_CONFIGS)
    
    # Crea heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=config['color'],
                cbar_kws={'label': config['label']}, ax=ax, linewidths=1)
    
    ax.set_xlabel('Numero di Parafrasi', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epoca', fontsize=12, fontweight='bold')
    ax.set_title(f'{method}: Convergenza {config["label"]}',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / f'{method}_{metric}_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_delta_analysis(df, method, metric, output_dir):
    """
    Grafico delta: variazione percentuale rispetto a para0 per ogni epoca
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    method_data = df[df['trainer'] == method]
    config = METRICS_CONFIG[metric]
    
    # Calcola delta rispetto a para0 per ogni epoca
    for epoch in EPOCHS:
        epoch_data = method_data[method_data['epoch'] == epoch]
        baseline = epoch_data[epoch_data['num_paraphrases'] == 0][metric].values[0]
        
        deltas = []
        for para_count in PARAPHRASE_CONFIGS[1:]:  # Escludi para0
            para_value = epoch_data[epoch_data['num_paraphrases'] == para_count][metric].values[0]
            delta_pct = ((para_value - baseline) / baseline) * 100 if baseline != 0 else 0
            deltas.append(delta_pct)
        
        ax.plot(PARAPHRASE_CONFIGS[1:], deltas, marker='o', linewidth=2, 
                markersize=8, label=f'Epoca {epoch}')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Numero di Parafrasi', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variazione % rispetto a para0', fontsize=12, fontweight='bold')
    ax.set_title(f'{method}: Impatto Parafrasi su {config["label"]}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{method}_{metric}_delta.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def generate_summary_report(df, output_dir):
    """Genera report testuale con statistiche chiave"""
    report_path = output_dir / 'convergence_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ANALISI CONVERGENZA E DINAMICHE TEMPORALI\n")
        f.write("=" * 80 + "\n\n")
        
        for method in METHODS:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"METODO: {method}\n")
            f.write(f"{'=' * 80}\n")
            
            method_data = df[df['trainer'] == method]
            
            # Analisi Exact Memorization
            f.write("\n--- Exact Memorization ---\n")
            for para in PARAPHRASE_CONFIGS:
                para_data = method_data[method_data['num_paraphrases'] == para]
                values = para_data.sort_values('epoch')['exact_memorization'].values
                if len(values) >= 2:
                    improvement = ((values[0] - values[-1]) / values[0]) * 100
                    f.write(f"  para{para}: {values[0]:.3f} (ep5) → {values[-1]:.3f} (ep20) "
                           f"[{improvement:+.1f}%]\n")
            
            # Analisi Model Utility
            f.write("\n--- Model Utility ---\n")
            for para in PARAPHRASE_CONFIGS:
                para_data = method_data[method_data['num_paraphrases'] == para]
                values = para_data.sort_values('epoch')['model_utility'].values
                if len(values) >= 2:
                    change = ((values[-1] - values[0]) / values[0]) * 100
                    f.write(f"  para{para}: {values[0]:.3f} (ep5) → {values[-1]:.3f} (ep20) "
                           f"[{change:+.1f}%]\n")
            
            # Velocità di convergenza
            f.write("\n--- Velocità Convergenza (EM) ---\n")
            for para in PARAPHRASE_CONFIGS:
                para_data = method_data[method_data['num_paraphrases'] == para].sort_values('epoch')
                if len(para_data) >= 2:
                    em_values = para_data['exact_memorization'].values
                    # Calcola quando raggiunge il 90% del miglioramento totale
                    total_improvement = em_values[0] - em_values[-1]
                    if total_improvement > 0.01:
                        target_value = em_values[0] - (0.9 * total_improvement)
                        converge_epoch = None
                        for idx, (ep, val) in enumerate(zip(para_data['epoch'], em_values)):
                            if val <= target_value:
                                converge_epoch = ep
                                break
                        if converge_epoch:
                            f.write(f"  para{para}: Raggiunge 90% miglioramento a epoca {converge_epoch}\n")
                    else:
                        f.write(f"  para{para}: Nessun miglioramento significativo\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINE REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analizza evoluzione metriche attraverso epoche')
    parser.add_argument('--base-dir', type=str, 
                       default='saves/unlearn',
                       help='Directory base con i checkpoint')
    parser.add_argument('--output', type=str,
                       default='plots/epoch_evolution',
                       help='Directory output per i grafici')
    parser.add_argument('--methods', nargs='+', default=METHODS,
                       help='Metodi da analizzare')
    parser.add_argument('--metrics', nargs='+', 
                       default=['exact_memorization', 'model_utility', 'mia_loss', 
                               'forget_Q_A_gibberish', 'privleak'],
                       help='Metriche da analizzare')
    parser.add_argument('--epochs', nargs='+', type=int, default=EPOCHS,
                       help='Epoche da analizzare')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ANALISI EVOLUZIONE TEMPORALE METRICHE")
    print("=" * 80)
    print(f"Base dir: {args.base_dir}")
    print(f"Output: {output_dir}")
    print(f"Metodi: {args.methods}")
    print(f"Metriche: {args.metrics}")
    print(f"Epoche: {args.epochs}")
    print("=" * 80 + "\n")
    
    # Carica dati direttamente dai checkpoint
    df = load_data(args.base_dir, args.epochs)
    
    if df.empty:
        print("\n❌ No data loaded! Exiting.")
        return
    
    # Genera grafici per ogni metodo e metrica
    for method in args.methods:
        print(f"\n→ Processando {method}...")
        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)
        
        for metric in args.metrics:
            if metric not in df.columns:
                print(f"  ⚠ Metrica '{metric}' non trovata, skip")
                continue
            
            # Grafico evoluzione
            plot_method_evolution(df, method, metric, method_dir)
            
            # Heatmap convergenza
            plot_convergence_heatmap(df, method, metric, method_dir)
            
            # Analisi delta
            plot_delta_analysis(df, method, metric, method_dir)
    
    # Grafici comparativi tra metodi
    print("\n→ Generando confronti tra metodi...")
    comparison_dir = output_dir / 'comparisons'
    comparison_dir.mkdir(exist_ok=True)
    
    for metric in args.metrics:
        for epoch in EPOCHS:
            plot_all_methods_comparison(df, metric, epoch, comparison_dir)
    
    # Report testuale
    generate_summary_report(df, output_dir)
    
    # Salva CSV aggiornato con i dati corretti
    csv_path = output_dir / 'summary_all_detailed_metrics.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✓ CSV aggiornato salvato: {csv_path}")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETATO!")
    print("=" * 80)
    print(f"\nGrafici salvati in: {output_dir}")
    print(f"  - Per metodo: {output_dir}/[METODO]/")
    print(f"  - Confronti: {output_dir}/comparisons/")
    print(f"  - Report: {output_dir}/convergence_summary.txt")
    print(f"  - CSV: {output_dir}/summary_all_detailed_metrics.csv")


if __name__ == '__main__':
    main()
