#!/usr/bin/env python3
"""
Script per generare grafici a linee per ogni metrica individuale di ogni metodo di unlearning.
Analizza l'effetto del numero di parafrasi su tutte le metriche mostrando chiaramente il trend.

Genera:
- Grafici a linee per ogni metrica di ogni metodo
- Organizzati in cartelle per metodo
- Grafici comparativi tra metodi
- CSV riassuntivo con tutte le metriche

Usage:
    python scripts/plot/plot_line_metrics.py
    python scripts/plot/plot_line_metrics.py --paraphrases 0 5 10 15 20
    python scripts/plot/plot_line_metrics.py --output-dir plots/line_metrics
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

# Metriche aggregate da escludere
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

HIGHER_IS_BETTER = [
    "forget_Q_A_gibberish",
    "model_utility",
]

SPECIAL_METRICS = {
    "privleak": "closer_to_zero",
    "mia_loss": "closer_to_half",
    "mia_min_k": "closer_to_half",
    "mia_min_k_plus_plus": "closer_to_half",
    "mia_zlib": "closer_to_half",
}

# Gruppi di metriche per i grafici compositi
METRIC_GROUPS = [
    {
        "name": "Memorizzazione Letterale",
        "filename": "memorizzazione_letterale",
        "metrics": ["exact_memorization", "extraction_strength"],
    },
    {
        "name": "Memorizzazione Semantica",
        "filename": "memorizzazione_semantica",
        "metrics": ["forget_Q_A_Prob", "forget_Q_A_PARA_Prob", "forget_truth_ratio"],
    },
    {
        "name": "Privacy (MIA)",
        "filename": "privacy_mia",
        "metrics": ["mia_loss", "mia_min_k_plus_plus", "privleak"],
    },
    {
        "name": "Utilità e Forget Fluency",
        "filename": "utilita_forget_fluency",
        "metrics": ["model_utility", "forget_Q_A_gibberish"],
    },
]


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
    """
    results = []

    for trainer in TRAINERS:
        for num_para in paraphrase_counts:
            pattern = f"{base_dir}/tofu_Llama-3.2-1B-Instruct_forget10_{trainer}_para{num_para}_ep20"
            exp_dirs = glob.glob(pattern)

            if not exp_dirs:
                print(f"⚠️  Missing: {trainer} with {num_para} paraphrases")
                continue

            exp_dir = exp_dirs[0]

            checkpoint_dirs = []
            for ckpt_dir in glob.glob(f"{exp_dir}/checkpoint-*"):
                eval_file = os.path.join(ckpt_dir, "evals", "TOFU_SUMMARY.json")
                if os.path.exists(eval_file):
                    step = extract_checkpoint_number(ckpt_dir)
                    checkpoint_dirs.append((step, ckpt_dir, eval_file))

            if not checkpoint_dirs:
                print(f"⚠️  No evaluated checkpoints: {trainer} para={num_para}")
                continue

            checkpoint_dirs.sort(key=lambda x: x[0])
            max_step = checkpoint_dirs[-1][0]
            steps_per_epoch = max_step / 20.0

            for target_epoch in target_epochs:
                target_step = int(target_epoch * steps_per_epoch)

                closest_ckpt = min(
                    checkpoint_dirs, key=lambda x: abs(x[0] - target_step)
                )
                actual_step, ckpt_dir, summary_file = closest_ckpt
                actual_epoch = round(actual_step / steps_per_epoch)

                if abs(actual_epoch - target_epoch) > 0.5:
                    print(
                        f"⚠️  No checkpoint near epoch {target_epoch}: {trainer} para={num_para}"
                    )
                    continue

                try:
                    with open(summary_file, "r") as f:
                        data = json.load(f)

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
                        f"✅ Loaded: {trainer} para={num_para} epoch={target_epoch} "
                        f"(step={actual_step}, "
                        f"{len([k for k in data.keys() if k not in AGGREGATE_METRICS])} metrics)"
                    )

                except Exception as e:
                    print(f"❌ Error loading {summary_file}: {e}")

    return pd.DataFrame(results)


def _direction_label(metric: str) -> str:
    """Restituisce l'etichetta della direzione ottimale per una metrica."""
    if metric in SPECIAL_METRICS:
        if SPECIAL_METRICS[metric] == "closer_to_zero":
            return "≈ closer to 0 is better"
        elif SPECIAL_METRICS[metric] == "closer_to_half":
            return "≈ closer to 0.5 is better"
    elif metric in HIGHER_IS_BETTER:
        return "↑ higher is better"
    return "↓ lower is better"


def plot_metric_line(
    df: pd.DataFrame, trainer: str, metric: str, epoch: int, output_dir: str
):
    """
    Genera grafico a linee per una metrica di un metodo al variare delle parafrasi.
    """
    trainer_dir = os.path.join(output_dir, trainer, f"epoch_{epoch}")
    os.makedirs(trainer_dir, exist_ok=True)

    metric_df = df[
        (df["trainer"] == trainer) & (df["metric"] == metric) & (df["epoch"] == epoch)
    ].sort_values("num_paraphrases")

    if metric_df.empty:
        return

    metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        metric_df["num_paraphrases"],
        metric_df["value"],
        marker="o",
        linewidth=2.2,
        markersize=8,
        color="steelblue",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    # Annota i valori sui punti
    for _, row in metric_df.iterrows():
        ax.annotate(
            f"{row['value']:.4f}",
            xy=(row["num_paraphrases"], row["value"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=13, fontweight="bold")
    ax.set_title(
        f"{metric_label} vs Paraphrases",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(metric_df["num_paraphrases"])

    ax.text(
        0.98,
        0.98,
        _direction_label(metric),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(trainer_dir, f"{safe_metric_name}_vs_paraphrases.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_comparison_across_trainers(
    df: pd.DataFrame, metric: str, epoch: int, output_dir: str
):
    """
    Genera grafico a linee comparativo tra tutti i metodi per una metrica specifica.
    Senza legenda individuale (usare i grafici compositi per la legenda condivisa).
    """
    comparison_dir = os.path.join(output_dir, "comparisons", f"epoch_{epoch}")
    os.makedirs(comparison_dir, exist_ok=True)

    metric_df = df[(df["metric"] == metric) & (df["epoch"] == epoch)]

    if metric_df.empty:
        return

    metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())
    colors = sns.color_palette("husl", len(TRAINERS))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, trainer in enumerate(TRAINERS):
        trainer_df = metric_df[metric_df["trainer"] == trainer].sort_values(
            "num_paraphrases"
        )
        if trainer_df.empty:
            continue

        ax.plot(
            trainer_df["num_paraphrases"],
            trainer_df["value"],
            marker=markers[idx % len(markers)],
            label=trainer,
            linewidth=2,
            markersize=7,
            color=colors[idx],
        )

    para_ticks = sorted(metric_df["num_paraphrases"].unique())
    ax.set_xticks(para_ticks)
    ax.set_xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=13, fontweight="bold")
    ax.set_title(
        f"{metric_label} vs Paraphrases",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.text(
        0.98,
        0.98,
        _direction_label(metric),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(comparison_dir, f"{safe_metric_name}_all_methods.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_composite_group(
    df: pd.DataFrame, group: dict, epoch: int, output_dir: str
):
    """
    Genera un'immagine composita con i subplot di un gruppo di metriche,
    con una singola legenda condivisa in alto al centro.
    """
    composite_dir = os.path.join(output_dir, "composite", f"epoch_{epoch}")
    os.makedirs(composite_dir, exist_ok=True)

    metrics = [m for m in group["metrics"] if not df[(df["metric"] == m) & (df["epoch"] == epoch)].empty]
    if not metrics:
        return

    n_metrics = len(metrics)
    colors = sns.color_palette("husl", len(TRAINERS))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(5.5 * n_metrics, 4.5),
        squeeze=False,
    )
    axes = axes[0]  # flatten to 1D

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        metric_df = df[(df["metric"] == metric) & (df["epoch"] == epoch)]
        metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())

        for t_idx, trainer in enumerate(TRAINERS):
            trainer_df = metric_df[metric_df["trainer"] == trainer].sort_values(
                "num_paraphrases"
            )
            if trainer_df.empty:
                continue

            ax.plot(
                trainer_df["num_paraphrases"],
                trainer_df["value"],
                marker=markers[t_idx % len(markers)],
                label=trainer,
                linewidth=2,
                markersize=7,
                color=colors[t_idx],
            )

        para_ticks = sorted(metric_df["num_paraphrases"].unique())
        ax.set_xticks(para_ticks)
        ax.set_xlabel("Number of Paraphrases", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
        ax.set_title(
            f"{metric_label}",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )

        ax.text(
            0.98, 0.02,
            _direction_label(metric),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.grid(True, alpha=0.3)

    # Legenda condivisa in alto al centro, presa dal primo asse
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(TRAINERS),
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        group["name"],
        fontsize=15,
        fontweight="bold",
        y=1.07,
    )

    plt.tight_layout()
    output_file = os.path.join(composite_dir, f"{group['filename']}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {group['name']} ({n_metrics} subplots)")


def plot_all_metrics_for_trainer(
    df: pd.DataFrame, trainer: str, epoch: int, output_dir: str
):
    """
    Genera un grafico a linee con TUTTE le metriche per un singolo metodo per una data epoca.
    Ogni linea rappresenta una metrica al variare delle parafrasi.
    """
    trainer_dir = os.path.join(output_dir, trainer, f"epoch_{epoch}")
    os.makedirs(trainer_dir, exist_ok=True)

    trainer_df = df[(df["trainer"] == trainer) & (df["epoch"] == epoch)]

    if trainer_df.empty:
        return

    all_metrics = sorted(trainer_df["metric"].unique())
    paraphrase_counts = sorted(trainer_df["num_paraphrases"].unique())
    colors = sns.color_palette("husl", len(all_metrics))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, metric in enumerate(all_metrics):
        metric_df = trainer_df[trainer_df["metric"] == metric].sort_values(
            "num_paraphrases"
        )
        if metric_df.empty:
            continue

        metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())
        ax.plot(
            metric_df["num_paraphrases"],
            metric_df["value"],
            marker=markers[idx % len(markers)],
            label=metric_label,
            linewidth=1.8,
            markersize=6,
            color=colors[idx],
        )

    ax.set_xticks(paraphrase_counts)
    ax.set_xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{trainer} – All Metrics vs Paraphrases",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(trainer_dir, "all_metrics_lines.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"   📊 {trainer} epoch {epoch}: all_metrics_lines.png")
    plt.close()


def plot_epochs_evolution_for_trainer(
    df: pd.DataFrame, trainer: str, metric: str, output_dir: str
):
    """
    Genera grafico a linee che mostra l'evoluzione di una metrica per diverse epoche,
    al variare delle parafrasi. Ogni linea rappresenta un'epoca.
    """
    trainer_dir = os.path.join(output_dir, trainer, "epoch_evolution")
    os.makedirs(trainer_dir, exist_ok=True)

    metric_df = df[(df["trainer"] == trainer) & (df["metric"] == metric)]

    if metric_df.empty:
        return

    all_epochs = sorted(metric_df["epoch"].unique())
    if len(all_epochs) < 2:
        return  # Non ha senso plottare evoluzione con una sola epoca

    metric_label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())
    colors = sns.color_palette("coolwarm", len(all_epochs))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, epoch in enumerate(all_epochs):
        epoch_df = metric_df[metric_df["epoch"] == epoch].sort_values("num_paraphrases")
        if epoch_df.empty:
            continue

        ax.plot(
            epoch_df["num_paraphrases"],
            epoch_df["value"],
            marker=markers[idx % len(markers)],
            label=f"Epoch {epoch}",
            linewidth=2,
            markersize=7,
            color=colors[idx],
        )

    para_ticks = sorted(metric_df["num_paraphrases"].unique())
    ax.set_xticks(para_ticks)
    ax.set_xlabel("Number of Paraphrases", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=13, fontweight="bold")
    ax.set_title(
        f"{metric_label} vs Paraphrases",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="best", fontsize=10)

    ax.text(
        0.98,
        0.98,
        _direction_label(metric),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(trainer_dir, f"{safe_metric_name}_epoch_evolution.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Genera tabella CSV con tutte le metriche dettagliate."""
    os.makedirs(output_dir, exist_ok=True)

    summary = df.pivot_table(
        index=["trainer", "epoch", "num_paraphrases"], columns="metric", values="value"
    ).reset_index()

    output_file = os.path.join(output_dir, "summary_all_detailed_metrics.csv")
    summary.to_csv(output_file, index=False, float_format="%.6f")
    print(f"\n📋 Saved: {output_file}")

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
        description="Generate line-chart metric plots for unlearning ablation study"
    )
    parser.add_argument(
        "--base-dir", default="saves/unlearn", help="Base directory for results"
    )
    parser.add_argument(
        "--output-dir",
        default="plots/ablation_study_detailed_new_figures",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--paraphrases",
        nargs="+",
        type=int,
        default=[0, 5, 10, 15, 20],
        help="Paraphrase counts to analyze",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Epochs to analyze",
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Skip generating comparison plots across trainers",
    )
    parser.add_argument(
        "--skip-epoch-evolution",
        action="store_true",
        help="Skip generating epoch evolution plots",
    )

    args = parser.parse_args()

    print("\n" + "=" * 120)
    print("TOFU Ablation Study - Line Chart Metrics Analysis")
    print("=" * 120 + "\n")

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

    all_metrics = sorted(df["metric"].unique())
    print(f"\n📊 Metrics to analyze ({len(all_metrics)}):")
    for metric in all_metrics:
        print(f"   - {metric}")

    # Grafici a linee individuali per ogni metodo, metrica ed epoca
    print("\n📈 Generating individual line charts by method and epoch...\n")
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
                plot_metric_line(df, trainer, metric, epoch, args.output_dir)

    # Grafico a linee con TUTTE le metriche per ogni metodo ed epoca
    print("\n📈 Generating all-metrics line plots per method and epoch...\n")
    for epoch in sorted(df["epoch"].unique()):
        print(f"\n   === Epoch {epoch} ===")
        for trainer in TRAINERS:
            trainer_df = df[(df["trainer"] == trainer) & (df["epoch"] == epoch)]
            if not trainer_df.empty:
                plot_all_metrics_for_trainer(df, trainer, epoch, args.output_dir)

    # Grafici comparativi tra metodi (senza legenda individuale)
    if not args.skip_comparisons:
        print("\n📈 Generating comparison line charts across methods...\n")
        for epoch in sorted(df["epoch"].unique()):
            print(f"\n   === Epoch {epoch} ===")
            for metric in all_metrics:
                plot_metric_comparison_across_trainers(
                    df, metric, epoch, args.output_dir
                )
                print(f"   ✅ {metric}")

    # Grafici compositi per gruppo con legenda condivisa
    print("\n📈 Generating composite group plots with shared legend...\n")
    for epoch in sorted(df["epoch"].unique()):
        print(f"\n   === Epoch {epoch} ===")
        for group in METRIC_GROUPS:
            plot_composite_group(df, group, epoch, args.output_dir)

    # Evoluzione per epoche (solo se ci sono più epoche)
    if not args.skip_epoch_evolution and len(sorted(df["epoch"].unique())) > 1:
        print("\n📈 Generating epoch evolution line charts...\n")
        for trainer in TRAINERS:
            trainer_df = df[df["trainer"] == trainer]
            if trainer_df.empty:
                continue
            print(f"   {trainer}:")
            for metric in all_metrics:
                plot_epochs_evolution_for_trainer(df, trainer, metric, args.output_dir)
                print(f"      ✅ {metric}")

    # Tabella riassuntiva
    print("\n📋 Generating summary table...")
    generate_summary_table(df, args.output_dir)

    print("\n" + "=" * 120)
    print(f"✅ All plots and tables saved in: {args.output_dir}/")
    print("\n   Structure:")
    print(f"   - {args.output_dir}/[METHOD]/epoch_[N]/*_vs_paraphrases.png  (individual line charts)")
    print(f"   - {args.output_dir}/[METHOD]/epoch_[N]/all_metrics_lines.png  (all metrics overlay)")
    print(f"   - {args.output_dir}/[METHOD]/epoch_evolution/*_epoch_evolution.png  (epoch trends)")
    print(f"   - {args.output_dir}/comparisons/epoch_[N]/*_all_methods.png  (comparison plots)")
    print(f"   - {args.output_dir}/composite/epoch_[N]/*.png  (composite group plots, shared legend)")
    print(f"   - {args.output_dir}/summary_all_detailed_metrics.csv  (full data table)")
    print(f"\n   Epochs analyzed: {sorted(df['epoch'].unique())}")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
