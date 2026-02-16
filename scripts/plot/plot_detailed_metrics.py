#!/usr/bin/env python3
"""
Script per generare istogrammi dettagliati per ogni metrica individuale di ogni metodo di unlearning.
Analizza l'effetto del numero di parafrasi su tutte le metriche (non solo quelle aggregate).

Genera:
- Istogrammi per ogni metrica di ogni metodo
- Organizzati in cartelle per metodo
- CSV riassuntivo con tutte le metriche

Usage:
    python scripts/plot/plot_detailed_metrics.py
    python scripts/plot/plot_detailed_metrics.py --paraphrases 0 5 10 15 20
    python scripts/plot/plot_detailed_metrics.py --output-dir plots/detailed_metrics
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
import numpy as np

# Configurazione stile grafici
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# Metriche aggregate da escludere (già analizzate nell'altro script)
AGGREGATE_METRICS = [
    "memorization_score",
    "privacy_score",
    "utility_score",
    "aggregate_score",
]

TRAINERS = ["GradDiff", "NPO", "SimNPO", "DPO", "RMU", "UNDIAL"]

# Mapping nomi metriche human-readable
METRIC_NAMES = {
    "exact_memorization": "Exact Memorization",
    "extraction_strength": "Extraction Strength",
    "forget_Q_A_PARA_Prob": "Forget Q+A Paraphrase Probability",
    "forget_Q_A_Prob": "Forget Q+A Probability",
    "forget_Q_A_ROUGE": "Forget Q+A ROUGE Score",
    "forget_Q_A_gibberish": "Forget Q+A Gibberish",
    "forget_quality": "Forget Quality",
    "forget_truth_ratio": "Forget Truth Ratio",
    "mia_loss": "MIA Loss",
    "mia_min_k": "MIA Min-K",
    "mia_min_k_plus_plus": "MIA Min-K++",
    "mia_zlib": "MIA Zlib",
    "model_utility": "Model Utility",
    "privleak": "Privacy Leakage",
}

# Metriche dove valori ALTI sono migliori (per colorazione)
# Basato sull'analisi teorica:
# - Memorization metrics: lower is better (vogliamo dimenticare)
# - Privacy metrics (MIA): closer to 0.5 is better (indistinguibilità statistica)
# - Utility metrics: higher is better (vogliamo preservare utilità)
HIGHER_IS_BETTER = [
    "forget_Q_A_gibberish",  # Fluency: higher = più fluente (meglio)
    "model_utility",  # Utility: higher = preserva più conoscenza (meglio)
]

# Metriche speciali (trattamento diverso)
SPECIAL_METRICS = {
    "privleak": "closer_to_zero",  # Closer to 0 is better (può essere +/-)
    "mia_loss": "closer_to_half",  # Closer to 0.5 = indistinguibilità ottimale
    "mia_min_k": "closer_to_half",  # Closer to 0.5 = indistinguibilità ottimale
    "mia_min_k_plus_plus": "closer_to_half",  # Closer to 0.5 = indistinguibilità ottimale
    "mia_zlib": "closer_to_half",  # Closer to 0.5 = indistinguibilità ottimale
}


def extract_checkpoint_number(path: str) -> int:
    """Estrae il numero di step dal checkpoint path."""
    match = re.search(r"checkpoint-(\d+)", path)
    return int(match.group(1)) if match else 0


def load_all_metrics(
    base_dir: str = "saves/unlearn",
    paraphrase_counts: List[int] = [0, 5, 10, 15, 20],
    target_epochs: List[int] = [20],
) -> pd.DataFrame:
    """
    Carica TUTTE le metriche dai checkpoint corrispondenti alle epoche specificate.

    Args:
        base_dir: Directory base dei risultati
        paraphrase_counts: Lista di configurazioni parafrasi
        target_epochs: Lista di epoche da analizzare (es: [5, 10, 15, 20])

    Returns:
        DataFrame con colonne: trainer, num_paraphrases, epoch, metric, value
    """
    results = []

    for trainer in TRAINERS:
        for num_para in paraphrase_counts:
            # Pattern experiment directory
            pattern = f"{base_dir}/tofu_Llama-3.2-1B-Instruct_forget10_{trainer}_para{num_para}_ep20"
            exp_dirs = glob.glob(pattern)

            if not exp_dirs:
                print(f"⚠️  Missing: {trainer} with {num_para} paraphrases")
                continue

            exp_dir = exp_dirs[0]

            # Trova tutti i checkpoint con eval
            checkpoint_dirs = []
            for ckpt_dir in glob.glob(f"{exp_dir}/checkpoint-*"):
                eval_file = os.path.join(ckpt_dir, "evals", "TOFU_SUMMARY.json")
                if os.path.exists(eval_file):
                    step = extract_checkpoint_number(ckpt_dir)
                    checkpoint_dirs.append((step, ckpt_dir, eval_file))

            if not checkpoint_dirs:
                print(f"⚠️  No evaluated checkpoints: {trainer} para={num_para}")
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
                closest_ckpt = min(
                    checkpoint_dirs, key=lambda x: abs(x[0] - target_step)
                )
                actual_step, ckpt_dir, summary_file = closest_ckpt
                actual_epoch = round(actual_step / steps_per_epoch)

                # Salta se l'epoca effettiva non corrisponde (tolleranza ±0.5 epoche)
                if abs(actual_epoch - target_epoch) > 0.5:
                    print(
                        f"⚠️  No checkpoint near epoch {target_epoch}: {trainer} para={num_para}"
                    )
                    continue

                try:
                    with open(summary_file, "r") as f:
                        data = json.load(f)

                    # Estrai TUTTE le metriche (escluse quelle aggregate)
                    for metric, value in data.items():
                        if metric not in AGGREGATE_METRICS:
                            results.append(
                                {
                                    "trainer": trainer,
                                    "num_paraphrases": num_para,
                                    "epoch": target_epoch,
                                    "metric": metric,
                                    "value": value,
                                }
                            )

                    print(
                        f"✅ Loaded: {trainer} para={num_para} epoch={target_epoch} (step={actual_step}, {len([k for k in data.keys() if k not in AGGREGATE_METRICS])} individual metrics)"
                    )

                except Exception as e:
                    print(f"❌ Error loading {summary_file}: {e}")

    return pd.DataFrame(results)


def plot_metric_histogram(
    df: pd.DataFrame, trainer: str, metric: str, epoch: int, output_dir: str
):
    """
    Genera istogramma per una metrica di un metodo al variare delle parafrasi per una specifica epoca.
    """
    trainer_dir = os.path.join(output_dir, trainer, f"epoch_{epoch}")
    os.makedirs(trainer_dir, exist_ok=True)

    metric_df = df[
        (df["trainer"] == trainer) & (df["metric"] == metric) & (df["epoch"] == epoch)
    ]

    if metric_df.empty:
        return

    # Ordina per numero parafrasi
    metric_df = metric_df.sort_values("num_paraphrases")
    values = metric_df["value"].values

    # Determina colorazione
    if metric in SPECIAL_METRICS:
        # Caso speciale: privleak (closer to 0 is better)
        if SPECIAL_METRICS[metric] == "closer_to_zero":
            abs_values = np.abs(values)
            vmin, vmax = abs_values.min(), abs_values.max()
            if vmax > vmin:
                # Inverti: valori assoluti bassi = verde, alti = rosso
                normalized = 1 - (abs_values - vmin) / (vmax - vmin)
            else:
                normalized = np.ones_like(values) * 0.5
            colors = plt.cm.RdYlGn(normalized)
        # Caso speciale: MIA metrics (closer to 0.5 is better)
        elif SPECIAL_METRICS[metric] == "closer_to_half":
            # Distanza da 0.5: 0 = ottimo (verde), > 0 = peggio (rosso)
            distance_from_half = np.abs(values - 0.5)
            vmin, vmax = distance_from_half.min(), distance_from_half.max()
            if vmax > vmin:
                # Inverti: distanza piccola = verde, distanza grande = rosso
                normalized = 1 - (distance_from_half - vmin) / (vmax - vmin)
            else:
                normalized = np.ones_like(values) * 0.5
            colors = plt.cm.RdYlGn(normalized)
    elif metric in HIGHER_IS_BETTER:
        # Normalizza valori per colorazione (0=rosso, 1=verde)
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.ones_like(values) * 0.5
        colors = plt.cm.RdYlGn(normalized)
    else:
        # Inverti (alto=rosso, basso=verde) - lower is better
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            normalized = 1 - (values - vmin) / (vmax - vmin)
        else:
            normalized = np.ones_like(values) * 0.5
        colors = plt.cm.RdYlGn(normalized)

    plt.figure(figsize=(6, 6))

    bars = plt.bar(
        metric_df["num_paraphrases"].astype(str),
        metric_df["value"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Aggiungi valori sopra le barre
    for bar, value in zip(bars, metric_df["value"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())

    plt.xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    plt.ylabel(metric_label, fontsize=13, fontweight="bold")
    plt.title(
        f"{trainer} (Epoch {epoch}): {metric_label} vs Paraphrases",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Aggiungi indicazione direzione migliore
    if metric in SPECIAL_METRICS:
        direction = "≈ closer to 0 is better"
    elif metric in HIGHER_IS_BETTER:
        direction = "↑ higher is better"
    else:
        direction = "↓ lower is better"
    plt.text(
        0.98,
        0.98,
        direction,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # Sanitizza nome file
    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    output_file = f"{trainer_dir}/{safe_metric_name}_vs_paraphrases.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_comparison_across_trainers(
    df: pd.DataFrame, metric: str, epoch: int, output_dir: str
):
    """
    Genera grafico comparativo tra tutti i metodi per una metrica specifica per una data epoca.
    """
    comparison_dir = os.path.join(output_dir, "comparisons", f"epoch_{epoch}")
    os.makedirs(comparison_dir, exist_ok=True)

    metric_df = df[(df["metric"] == metric) & (df["epoch"] == epoch)]

    if metric_df.empty:
        return

    plt.figure(figsize=(6, 6))

    # Usa colori distinti per ogni trainer
    colors = sns.color_palette("husl", len(TRAINERS))

    x = np.arange(len(metric_df["num_paraphrases"].unique()))
    width = 0.13  # Larghezza barre

    for idx, trainer in enumerate(TRAINERS):
        trainer_df = metric_df[metric_df["trainer"] == trainer].sort_values(
            "num_paraphrases"
        )
        if not trainer_df.empty:
            offset = (idx - len(TRAINERS) / 2 + 0.5) * width
            plt.bar(
                x + offset,
                trainer_df["value"],
                width,
                label=trainer,
                color=colors[idx],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
            )

    metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())

    plt.xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    plt.ylabel(metric_label, fontsize=13, fontweight="bold")
    plt.title(
        f"All Methods (Epoch {epoch}): {metric_label} vs Paraphrases",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(x, sorted(metric_df["num_paraphrases"].unique()))
    plt.legend(loc="best", fontsize=10, ncol=2)

    # Aggiungi indicazione direzione migliore
    if metric in SPECIAL_METRICS:
        if SPECIAL_METRICS[metric] == "closer_to_zero":
            direction = "≈ closer to 0 is better"
        elif SPECIAL_METRICS[metric] == "closer_to_half":
            direction = "≈ closer to 0.5 is better"
        else:
            direction = "special metric"
    elif metric in HIGHER_IS_BETTER:
        direction = "↑ higher is better"
    else:
        direction = "↓ lower is better"
    plt.text(
        0.98,
        0.98,
        direction,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    output_file = f"{comparison_dir}/{safe_metric_name}_all_methods.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metrics_for_trainer(
    df: pd.DataFrame, trainer: str, epoch: int, output_dir: str
):
    """
    Genera un grafico con TUTTE le metriche per un singolo metodo per una data epoca.
    Barre raggruppate per paraphrase count.
    """
    trainer_dir = os.path.join(output_dir, trainer, f"epoch_{epoch}")
    os.makedirs(trainer_dir, exist_ok=True)

    trainer_df = df[(df["trainer"] == trainer) & (df["epoch"] == epoch)]

    if trainer_df.empty:
        return

    # Ottieni lista ordinata di metriche e paraphrases
    all_metrics = sorted(trainer_df["metric"].unique())
    paraphrase_counts = sorted(trainer_df["num_paraphrases"].unique())

    # Crea figura larga per contenere tutte le metriche
    fig, ax = plt.subplots(figsize=(18, 8))

    # Prepara dati per grouped bar chart
    x = np.arange(len(all_metrics))
    width = 0.15  # Larghezza di ogni barra
    colors = sns.color_palette("husl", len(paraphrase_counts))

    for idx, num_para in enumerate(paraphrase_counts):
        para_df = trainer_df[trainer_df["num_paraphrases"] == num_para]

        # Ordina per metriche
        values = []
        for metric in all_metrics:
            metric_value = para_df[para_df["metric"] == metric]["value"].values
            values.append(metric_value[0] if len(metric_value) > 0 else 0)

        offset = (idx - len(paraphrase_counts) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=f"{num_para} paraphrases",
            color=colors[idx],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85,
        )

    # Formatta asse X con nomi metriche abbreviati
    metric_labels = [
        METRIC_NAMES.get(m, m.replace("_", " ").title()) for m in all_metrics
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{trainer} (Epoch {epoch}): All Metrics vs Paraphrases",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", fontsize=10, ncol=len(paraphrase_counts))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_file = f"{trainer_dir}/all_metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"   📊 {trainer} epoch {epoch}: all_metrics_comparison.png")
    plt.close()


def plot_all_metrics_heatmap(
    df: pd.DataFrame, trainer: str, epoch: int, output_dir: str
):
    """
    Genera una heatmap con TUTTE le metriche per un singolo metodo per una data epoca.
    Righe: metriche, Colonne: paraphrase counts
    """
    trainer_dir = os.path.join(output_dir, trainer, f"epoch_{epoch}")
    os.makedirs(trainer_dir, exist_ok=True)

    trainer_df = df[(df["trainer"] == trainer) & (df["epoch"] == epoch)]

    if trainer_df.empty:
        return

    # Pivot per heatmap: righe=metriche, colonne=num_paraphrases
    pivot_df = trainer_df.pivot(
        index="metric", columns="num_paraphrases", values="value"
    )

    # Ordina metriche per categoria
    memorization_metrics = [
        m
        for m in pivot_df.index
        if m
        in [
            "exact_memorization",
            "extraction_strength",
            "forget_Q_A_PARA_Prob",
            "forget_Q_A_Prob",
            "forget_Q_A_ROUGE",
            "forget_truth_ratio",
        ]
    ]
    privacy_metrics = [
        m
        for m in pivot_df.index
        if m
        in [
            "mia_loss",
            "mia_min_k",
            "mia_min_k_plus_plus",
            "mia_zlib",
            "forget_quality",
            "privleak",
        ]
    ]
    utility_metrics = [
        m for m in pivot_df.index if m in ["model_utility", "forget_Q_A_gibberish"]
    ]

    ordered_metrics = memorization_metrics + privacy_metrics + utility_metrics
    pivot_df = pivot_df.reindex(ordered_metrics)

    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 12))

    # Normalizza per row per migliore visualizzazione
    pivot_normalized = pivot_df.div(pivot_df.max(axis=1), axis=0)

    # Crea heatmap
    sns.heatmap(
        pivot_normalized,
        annot=pivot_df,  # Mostra valori originali
        fmt=".4f",
        cmap="RdYlGn",
        center=0.5,
        cbar_kws={"label": "Normalized Value (0-1)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Formatta labels
    metric_labels = [
        METRIC_NAMES.get(m, m.replace("_", " ").title()) for m in pivot_df.index
    ]
    ax.set_yticklabels(metric_labels, rotation=0, fontsize=10)
    ax.set_xticklabels([f"{int(c)}" for c in pivot_df.columns], rotation=0, fontsize=11)

    ax.set_xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metrics", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{trainer} (Epoch {epoch}): Metrics Heatmap (normalized by row)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    output_file = f"{trainer_dir}/all_metrics_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"   📊 {trainer} epoch {epoch}: all_metrics_heatmap.png")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Genera tabella CSV con tutte le metriche dettagliate."""
    os.makedirs(output_dir, exist_ok=True)

    # Pivot: righe=(trainer, epoch, num_paraphrases), colonne=metrics
    summary = df.pivot_table(
        index=["trainer", "epoch", "num_paraphrases"], columns="metric", values="value"
    ).reset_index()

    output_file = f"{output_dir}/summary_all_detailed_metrics.csv"
    summary.to_csv(output_file, index=False, float_format="%.6f")
    print(f"\n📋 Saved: {output_file}")

    # Stampa statistiche
    print("\n" + "=" * 120)
    print("SUMMARY: DETAILED METRICS BY METHOD, EPOCH AND PARAPHRASES")
    print("=" * 120)
    print(f"Total metrics analyzed: {len(df['metric'].unique())}")
    print(f"Methods: {', '.join(TRAINERS)}")
    print(f"Epochs: {sorted(df['epoch'].unique())}")
    print(f"Paraphrase configs: {sorted(df['num_paraphrases'].unique())}")
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed metric histograms for unlearning ablation study"
    )
    parser.add_argument(
        "--base-dir", default="saves/unlearn", help="Base directory for results"
    )
    parser.add_argument(
        "--output-dir",
        default="plots/ablation_study_detailed",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--paraphrases",
        nargs="+",
        type=int,
        default=[0, 5, 10, 15, 20],
        help="Paraphrase counts to analyze (0=baseline without paraphrases)",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Epochs to analyze (checkpoints at these epochs)",
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Skip generating comparison plots across trainers",
    )

    args = parser.parse_args()

    print("\n" + "=" * 120)
    print("TOFU Ablation Study - Detailed Metrics Analysis")
    print("=" * 120 + "\n")

    # Carica tutte le metriche
    print(f"📂 Loading all individual metrics for epochs: {args.epochs}...")
    df = load_all_metrics(args.base_dir, args.paraphrases, args.epochs)

    if df.empty:
        print("\n❌ No results found! Make sure to run the training first.")
        return

    print("\n✅ Loaded data:")
    print(f"   - Total data points: {len(df)}")
    print(f"   - Unique metrics: {len(df['metric'].unique())}")
    print(f"   - Trainers: {df['trainer'].nunique()}")
    print(f"   - Epochs: {sorted(df['epoch'].unique())}")
    print(f"   - Paraphrase configs: {sorted(df['num_paraphrases'].unique())}")

    # Lista metriche trovate
    all_metrics = sorted(df["metric"].unique())
    print(f"\n📊 Metrics to analyze ({len(all_metrics)}):")
    for metric in all_metrics:
        print(f"   - {metric}")

    # Genera istogrammi individuali per ogni metodo e epoca
    print("\n📊 Generating individual histograms by method and epoch...\n")

    for epoch in sorted(df["epoch"].unique()):
        print(f"\n   === Epoch {epoch} ===")
        for trainer in TRAINERS:
            trainer_df = df[(df["trainer"] == trainer) & (df["epoch"] == epoch)]
            if trainer_df.empty:
                print(f"   ⚠️  No data for {trainer} at epoch {epoch}")
                continue

            trainer_metrics = trainer_df["metric"].unique()
            print(f"   {trainer}: {len(trainer_metrics)} metrics")

            for metric in trainer_metrics:
                plot_metric_histogram(df, trainer, metric, epoch, args.output_dir)

    # Genera plot con TUTTE le metriche per ogni metodo e epoca
    print("\n📊 Generating all-metrics plots for each method and epoch...\n")

    for epoch in sorted(df["epoch"].unique()):
        print(f"\n   === Epoch {epoch} ===")
        for trainer in TRAINERS:
            trainer_df = df[(df["trainer"] == trainer) & (df["epoch"] == epoch)]
            if not trainer_df.empty:
                plot_all_metrics_for_trainer(df, trainer, epoch, args.output_dir)
                plot_all_metrics_heatmap(df, trainer, epoch, args.output_dir)

    # Genera grafici comparativi tra metodi per ogni epoca
    if not args.skip_comparisons:
        print("\n📊 Generating comparison plots across methods...\n")

        for epoch in sorted(df["epoch"].unique()):
            print(f"\n   === Epoch {epoch} ===")
            for metric in all_metrics:
                plot_metric_comparison_across_trainers(
                    df, metric, epoch, args.output_dir
                )
                print(f"   ✅ {metric}")

    # Genera tabella riassuntiva
    print("\n📋 Generating summary table...")
    generate_summary_table(df, args.output_dir)

    print("\n" + "=" * 120)
    print(f"✅ All plots and tables saved in: {args.output_dir}/")
    print("\n   Structure:")
    print(
        f"   - {args.output_dir}/[METHOD]/epoch_[N]/*_vs_paraphrases.png  (individual histograms)"
    )
    print(
        f"   - {args.output_dir}/[METHOD]/epoch_[N]/all_metrics_comparison.png  (grouped bar chart)"
    )
    print(
        f"   - {args.output_dir}/[METHOD]/epoch_[N]/all_metrics_heatmap.png  (heatmap)"
    )
    print(
        f"   - {args.output_dir}/comparisons/epoch_[N]/*_all_methods.png  (comparison plots)"
    )
    print(
        f"   - {args.output_dir}/summary_all_detailed_metrics.csv  (full data table with epochs)"
    )
    print(f"\n   Epochs analyzed: {sorted(df['epoch'].unique())}")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
