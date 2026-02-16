<div align="center">

<img src="assets/logo_s_unl.png" alt="S-UNL" width="200"/>

---

# Paraphrase Ablation Study
### Systematic Analysis of Linguistic Enrichment in LLM Unlearning

---

**An extension to the [OpenUnlearning](https://github.com/locuslab/open-unlearning) framework investigating how paraphrase enrichment affects machine unlearning effectiveness.**

[![Framework](https://img.shields.io/badge/Framework-OpenUnlearning-green)](https://github.com/locuslab/open-unlearning)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

</div>

---

## 📖 About This Study

This ablation study investigates the **impact of paraphrase enrichment on machine unlearning**, analyzing how linguistic variability affects:
- **Memorization removal** (Exact Memorization, Extraction Strength)
- **Privacy protection** (MIA attacks: Loss, Min-K, Min-K++, Zlib)
- **Model utility preservation** (retain set performance, forget fluency)

### Research Questions

1. **Does paraphrase enrichment improve unlearning?**  
   → **Yes**, especially for preference-based methods (DPO, NPO, SimNPO)

2. **What is the optimal number of paraphrases?**  
   → **5 paraphrases** provide ~70% of benefits; **20 paraphrases** achieve maximum performance

3. **What are the trade-offs?**  
   → Improved privacy (77-98% better) vs. slight utility degradation (5-10%)

4. **Which method benefits most?**  
   → **DPO** achieves exceptional privacy: MIA Min-K++ = **0.497** ≈ ideal 0.5

---

## 🎯 Key Results

| Method | Exact Mem. (para0→para20) | MIA Min-K++ | Model Utility | Best For |
|--------|---------------------------|-------------|---------------|----------|
| **DPO** | 0.062 → **0.005** (92%↓) | **0.497** ⭐ | 0.555 | Privacy-critical apps |
| **NPO** | 0.077 → **0.020** (74%↓) | 0.549 | 0.581 | Balanced scenarios |
| **SimNPO** | 0.213 → 0.074 (65%↓) | 0.683 | 0.565 | Good tradeoff |
| **RMU** | 0.042 → 0.022 (48%↓) | ~0.0 | **0.586** | Utility priority |
| **GradDiff** | 0.001 → 0.002 (~0%) | ~0.0 | 0.546 | Already minimal mem. |

**Key Insight**: MIA ≈ 0.5 means attacker **cannot distinguish** forget/holdout sets (optimal privacy). DPO achieves near-perfect indistinguishability.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.11+
- **GPU**: NVIDIA GPU with ≥16GB VRAM (tested on Ada Generation RTX 5000)
- **Storage**: ~50GB for all experiments
- **RAM**: ≥32GB recommended

### 1. Clone Repository

```bash
git clone https://github.com/agambalonga/S-UNL.git
cd open-unlearning
```

### 2. Environment Setup

```bash
# Create conda environment
conda create -n unlearning python=3.11
conda activate unlearning

# Install dependencies
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3
```

### 3. Data Setup

```bash
# Download evaluation logs and base models
python setup_data.py --eval

# This downloads:
# - TOFU dataset splits
# - Retain model evaluation logs
# - Base model references
# → Files saved to saves/eval/
```

### 4. Verify Installation

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check data
ls saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

---

## 🔬 Running the Ablation Study

### Configuration

The study is configured in [`scripts/paraphrases_ablation.sh`](scripts/paraphrases_ablation.sh):

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | Llama-3.2-1B-Instruct | 1.2B parameters, fits in 16GB GPU |
| **Dataset** | TOFU forget10 | 20 of 200 authors |
| **Methods** | GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL | 6 unlearning methods |
| **Paraphrases** | 0, 5, 10, 15, 20 | 0 = baseline, 20 = max enrichment |
| **Epochs** | 20 | Full training with checkpoints every epoch |
| **Batch size** | 4 (effective: 32) | 4 per device × 8 gradient accumulation |

### Launch Full Study

```bash
# Run all experiments (30 configurations: 6 methods × 5 paraphrase counts)
# Estimated time: 12-18 hours (includes automatic evaluation)
bash scripts/paraphrases_ablation.sh
```

**What happens:**
1. Trains 30 models (6 methods × 5 para configs)
2. **Automatically evaluates** each model at the end of every epoch
3. Saves checkpoints with evaluation results to `saves/unlearn/`
4. Logs everything to `logs/ablation_YYYYMMDD.log`

**No separate evaluation step needed!** All metrics are computed during training.

### Selective Training

Edit `scripts/paraphrases_ablation.sh` to run specific configurations:

```bash
# Train only DPO (fastest, ~2-3 hours per config)
trainers_order=("DPO")

# Test fewer paraphrase counts
paraphrase_counts=(0 5 20)  # Only baseline, para5, para20

# Train fewer epochs for testing
num_epochs=5
```

### Force Overwrite

```bash
# Re-run experiments, overwriting existing results
bash scripts/paraphrases_ablation.sh --force
```

---

## 📊 Results and Evaluation

### Automatic Evaluation During Training

The training script **automatically evaluates** each model at the end of every epoch thanks to the configuration:
- `trainer.args.eval_strategy=epoch`
- `trainer.args.eval_on_start=false`

**No separate evaluation step is needed!** Results are saved directly in checkpoint directories.

### Check Training Progress

```bash
# Watch training progress in real-time
tail -f logs/ablation_$(date +%Y%m%d).log

# Check completed evaluations
find saves/unlearn/tofu_*_para*_ep20/ -name "TOFU_SUMMARY.json" | wc -l
# Expected: 120 (30 models × 4 epochs with evals)
```

### View Results

```bash
# View summary of a specific evaluation
cat saves/unlearn/tofu_*_DPO_para20_ep20/checkpoint-80/evals/TOFU_SUMMARY.json | jq .

# Key metrics to check:
# - "exact_memorization": Should be ~0.005 for DPO para20
# - "mia_min_k_plus_plus": Should be ~0.497 for DPO para20 (near perfect!)
# - "model_utility": Should be ~0.555 for DPO para20
```

### Manual Re-Evaluation (Optional)

If you need to re-evaluate a specific checkpoint with different metrics:

```bash
# Example: Re-evaluate DPO with 20 paraphrases at epoch 20
model="Llama-3.2-1B-Instruct"
trainer="DPO"
num_para=20
checkpoint="saves/unlearn/tofu_${model}_forget10_${trainer}_para${num_para}_ep20/checkpoint-80"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    experiment=eval/tofu/paraphrase.yaml \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model=${model} \
    task_name=eval_${trainer}_para${num_para} \
    model.model_args.pretrained_model_name_or_path=${checkpoint} \
    paths.output_dir=${checkpoint}/evals \
    retain_logs_path=saves/eval/tofu_${model}_retain90/TOFU_EVAL.json
```

**Note:** This is rarely needed since evaluation happens automatically during training.

---

## 📈 Generate Plots

Visualize results across all methods and configurations.

### Detailed Metrics Plots

```bash
# Generate histograms for all individual metrics
python scripts/plot/plot_detailed_metrics.py \
    --epochs 5 10 15 20 \
    --paraphrases 0 5 10 15 20 \
    --output-dir plots/ablation_study_detailed

# Outputs:
# - plots/ablation_study_detailed/DPO/epoch_20/exact_memorization_vs_paraphrases.png
# - plots/ablation_study_detailed/DPO/epoch_20/mia_min_k_plus_plus_vs_paraphrases.png
# - plots/ablation_study_detailed/comparisons/epoch_20/mia_min_k_plus_plus_all_methods.png
# - plots/ablation_study_detailed/summary_all_detailed_metrics.csv
```

### Epoch Evolution Plots

```bash
# Show how metrics evolve during training
python scripts/plot/plot_epoch_evolution.py \
    --epochs 5 10 15 20 \
    --paraphrases 0 5 10 15 20 \
    --output-dir plots/epoch_evolution

# Outputs:
# - plots/epoch_evolution/preference_methods_em.png
# - plots/epoch_evolution/privacy_methods_mia_min_k_plus_plus.png
# - plots/epoch_evolution/utility_comparison.png
```

### View Plots

```bash
# Open comparison plot for MIA Min-K++
xdg-open plots/ablation_study_detailed/comparisons/epoch_20/mia_min_k_plus_plus_all_methods.png

# Or browse all plots
cd plots/ablation_study_detailed && firefox index.html
```

---

## 📁 Understanding Results Structure

```
open-unlearning/
├── saves/unlearn/
│   ├── tofu_Llama-3.2-1B-Instruct_forget10_DPO_para0_ep20/    # Baseline (no paraphrases)
│   │   ├── checkpoint-20/   # Epoch 5 (4 steps/epoch × 5)
│   │   ├── checkpoint-40/   # Epoch 10
│   │   ├── checkpoint-60/   # Epoch 15
│   │   └── checkpoint-80/   # Epoch 20 ⭐ Final
│   │       ├── evals/
│   │       │   ├── TOFU_SUMMARY.json  ← Key results here
│   │       │   └── TOFU_EVAL.json     ← Detailed outputs
│   │       ├── config.json
│   │       └── pytorch_model.bin
│   ├── tofu_Llama-3.2-1B-Instruct_forget10_DPO_para5_ep20/    # 5 paraphrases
│   ├── tofu_Llama-3.2-1B-Instruct_forget10_DPO_para10_ep20/   # 10 paraphrases
│   ├── tofu_Llama-3.2-1B-Instruct_forget10_DPO_para15_ep20/   # 15 paraphrases
│   ├── tofu_Llama-3.2-1B-Instruct_forget10_DPO_para20_ep20/   # 20 paraphrases
│   └── [25 more configurations for other methods...]
│
├── plots/
│   ├── ablation_study_detailed/
│   │   ├── DPO/epoch_20/
│   │   │   ├── exact_memorization_vs_paraphrases.png
│   │   │   ├── mia_min_k_plus_plus_vs_paraphrases.png
│   │   │   └── all_metrics_heatmap.png
│   │   ├── comparisons/epoch_20/
│   │   │   └── mia_min_k_plus_plus_all_methods.png  ← Compare all methods
│   │   └── summary_all_detailed_metrics.csv
│   └── epoch_evolution/
│       ├── preference_methods_em.png
│       └── privacy_methods_mia_min_k_plus_plus.png
│
└── logs/
    └── ablation_20260216.log  ← Full training log with timestamps
```

### Key Metrics in TOFU_SUMMARY.json

```json
{
  "exact_memorization": 0.005,           // Lower is better (0 = perfect)
  "extraction_strength": 0.012,          // Lower is better
  "mia_min_k_plus_plus": 0.497,         // Closer to 0.5 is better ⭐
  "mia_loss": 0.580,                     // Closer to 0.5 is better
  "mia_min_k": 0.561,                    // Closer to 0.5 is better
  "mia_zlib": 0.481,                     // Closer to 0.5 is better
  "model_utility": 0.555,                // Higher is better (0-1)
  "forget_Q_A_gibberish": 0.884,        // Higher = more fluent (0-1)
  "privleak": -28.9                      // Closer to 0 is better
}
```

---

## 🎓 Understanding the Experiment

### What are Paraphrases?

Instead of training only on original questions like:
```
Q: What is the capital of Panglossia?
A: I don't know.
```

We augment with semantically equivalent variations:
```
Original: What is the capital of Panglossia?
Para 1:   Can you tell me the capital city of Panglossia?
Para 2:   Which city serves as Panglossia's capital?
Para 3:   What city is the capital of Panglossia?
```

**Hypothesis**: Linguistic diversity helps models forget more robustly.

### Why DPO Uses "I don't know" (IDK)?

DPO is a preference-based method that learns from comparisons:
- **Preferred response**: "I don't know" (evasive)
- **Dispreferred response**: Original factual answer

This teaches the model to prefer evasive responses over revealing forgotten knowledge.

### Interpreting MIA Metrics

**MIA (Membership Inference Attack)** tries to guess if an example was in training:
- **AUC = 1.0**: Perfect attack (bad for privacy)
- **AUC = 0.5**: Random guessing (perfect privacy) ⭐
- **AUC = 0.0**: Inverted attack (also bad)

**Goal**: Get as close to 0.5 as possible.

**DPO achieves 0.497** = almost perfectly indistinguishable!

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA OOM error during training

**Solution**: Reduce batch size in `scripts/paraphrases_ablation.sh`:
```bash
per_device_train_batch_size=2  # Default: 4
gradient_accumulation_steps=16  # Default: 8 (keep effective batch size ~32)
```

### SSL Certificate Errors

**Symptoms**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**: Already handled in script via:
```bash
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0
```

If issues persist:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Missing Base Model

**Symptoms**: `Model 'open-unlearning/tofu_Llama-3.2-1B-Instruct_full' not found`

**Solution**: Download manually:
```bash
huggingface-cli login  # If model is gated
huggingface-cli download open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

Or specify local path in script:
```bash
model_path="/path/to/your/local/Llama-3.2-1B-Instruct"
```

### Evaluation Not Finding Checkpoints

**Symptoms**: No `TOFU_SUMMARY.json` files after training

**Solution**: Evaluation runs automatically during training. If files are missing:
```bash
# Check if checkpoints exist
ls saves/unlearn/tofu_*/checkpoint-*/

# Check if evaluation ran (look for "evals" folders)
find saves/unlearn/tofu_*_para*_ep20/checkpoint-*/ -name "evals" -type d

# If missing, training may have been interrupted
# Re-run training (it will skip completed models unless --force):
bash scripts/paraphrases_ablation.sh

# Or manually evaluate a specific checkpoint:
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    experiment=eval/tofu/paraphrase.yaml \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/[YOUR_MODEL]/checkpoint-80 \
    paths.output_dir=saves/unlearn/[YOUR_MODEL]/checkpoint-80/evals
```

### Plots Show Wrong Colors

**Symptoms**: MIA plots colored incorrectly (higher values are green)

**Solution**: Ensure you're using the updated plotting script with corrected `SPECIAL_METRICS`:
```bash
grep "closer_to_half" scripts/plot/plot_detailed_metrics.py
# Should show: "mia_*": "closer_to_half"
```

---

## 📊 Quick Analysis Commands

```bash
# Count completed training runs
ls -d saves/unlearn/tofu_*_para*_ep20/ | wc -l
# Expected: 30 (6 methods × 5 para configs)

# Check best DPO result
cat saves/unlearn/tofu_*_DPO_para20_ep20/checkpoint-80/evals/TOFU_SUMMARY.json | \
    jq '{exact_memorization, mia_min_k_plus_plus, model_utility}'

# Compare all methods at para20
for method in DPO NPO SimNPO GradDiff RMU UNDIAL; do
    echo "=== $method ==="
    cat saves/unlearn/tofu_*_${method}_para20_ep20/checkpoint-80/evals/TOFU_SUMMARY.json | \
        jq '{exact_memorization, mia_min_k_plus_plus, model_utility}'
done

# Generate CSV summary for all results
python scripts/plot/plot_detailed_metrics.py --epochs 20 --paraphrases 20
cat plots/ablation_study_detailed/summary_all_detailed_metrics.csv
```

---

## 🎯 Expected Results (Sanity Check)

After running the full study, verify key results:

### DPO para20 (Best Overall)
```bash
cat saves/unlearn/tofu_*_DPO_para20_ep20/checkpoint-80/evals/TOFU_SUMMARY.json
```

**Expected values** (±0.01):
- `exact_memorization`: **0.005** (excellent, 92% reduction from baseline)
- `mia_min_k_plus_plus`: **0.497** (near-perfect, 0.003 from ideal 0.5)
- `mia_loss`: **0.580** (good, within [0.45, 0.60])
- `model_utility`: **0.555** (acceptable, 5.7% reduction)

### Comparison Across Methods (para20)

| Method | EM | MIA Min-K++ | MU | Status |
|--------|----|--------------|----|--------|
| DPO | 0.005 | 0.497 | 0.555 | ✅ Best privacy |
| NPO | 0.020 | 0.549 | 0.581 | ✅ Good balance |
| SimNPO | 0.074 | 0.683 | 0.565 | ✅ Acceptable |
| RMU | 0.022 | ~0.0 | 0.586 | ⚠️ Poor privacy |
| GradDiff | 0.002 | ~0.0 | 0.546 | ⚠️ Poor privacy |

If your results significantly differ (>20%), check:
- Random seed (if set differently)
- Hyperparameters in script
- GPU memory constraints (may affect batch processing)

---

## 💡 Tips for Faster Iteration

### 1. Start Small
```bash
# Test with 5 epochs first (~30 min per method)
# Edit scripts/paraphrases_ablation.sh:
num_epochs=5

# Run single method
trainers_order=("DPO")

# Test fewer configs
paraphrase_counts=(0 20)  # Just baseline vs. maximum
```

### 2. Use Checkpoints
```bash
# If training interrupted, rerun with existing checkpoints
# Script automatically skips completed runs unless --force
bash scripts/paraphrases_ablation.sh
```

### 3. Parallel Training (Multiple GPUs)
```bash
# Terminal 1 (GPU 0): DPO, NPO, SimNPO
CUDA_VISIBLE_DEVICES=0 bash scripts/paraphrases_ablation.sh

# Terminal 2 (GPU 1): GradDiff, RMU, UNDIAL
# Edit script to change trainers_order, then:
CUDA_VISIBLE_DEVICES=1 bash scripts/paraphrases_ablation.sh
```

### 4. Monitor Progress
```bash
# Watch training log in real-time
tail -f logs/ablation_$(date +%Y%m%d).log

# Check GPU utilization
watch -n 1 nvidia-smi

# Count completed runs
watch -n 60 'ls -d saves/unlearn/tofu_*_para*/ | wc -l'
```

---

## 🔗 Related Resources

### OpenUnlearning Framework
- **Repository**: [github.com/locuslab/open-unlearning](https://github.com/locuslab/open-unlearning)
- **Paper**: [arXiv:2506.12618](https://arxiv.org/abs/2506.12618)
- **Models**: [HuggingFace Collection](https://huggingface.co/open-unlearning)

### Benchmarks & Datasets
- **TOFU**: [locuslab.github.io/tofu](https://locuslab.github.io/tofu/)
- **MUSE**: [muse-bench.github.io](https://muse-bench.github.io/)

### Unlearning Methods Implemented
- **DPO**: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- **NPO**: [Maini et al., 2024](https://arxiv.org/abs/2401.06121)
- **RMU**: [Li et al., 2024](https://arxiv.org/abs/2403.03218)
- **GradDiff**: [RFC 2024](https://arxiv.org/abs/2401.06121)

---

## 📝 Citation

If you use this ablation study or extend it, please cite both the OpenUnlearning framework and TOFU benchmark:

```bibtex
@article{openunlearning2025,
  title={{OpenUnlearning}: Accelerating {LLM} Unlearning via Unified Benchmarking},
  author={Dorna, Vineeth and Mekala, Anmol and others},
  journal={arXiv preprint arXiv:2506.12618},
  year={2025}
}

@inproceedings{maini2024tofu,
  title={{TOFU}: A Task of Fictitious Unlearning for {LLMs}},
  author={Maini, Pratyush and Feng, Zhili and others},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
```

---

## 🤝 Contributing

Found an issue or want to extend the study? Contributions welcome!

1. Fork the [OpenUnlearning repository](https://github.com/locuslab/open-unlearning)
2. Create a feature branch: `git checkout -b feature/paraphrase-extension`
3. Commit your changes: `git commit -am "Add new analysis"`
4. Push to branch: `git push origin feature/paraphrase-extension`
5. Submit a pull request

---

## 📄 License

This extension inherits the MIT License from OpenUnlearning. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built on [OpenUnlearning](https://github.com/locuslab/open-unlearning) • Maintained by the community**

Questions? [Open an issue](https://github.com/locuslab/open-unlearning/issues) or check the [main docs](docs/)

</div>



---

## 📊 Key Findings

| Method | Exact Mem. (para0→para20) | MIA Min-K++ | Model Utility | Best For |
|--------|---------------------------|-------------|---------------|----------|
| **DPO** | 0.062 → **0.005** (92%↓) | **0.497** ⭐ | 0.555 | Privacy-critical apps |
| **NPO** | 0.077 → **0.020** (74%↓) | 0.549 | 0.581 | Balanced scenarios |
| **SimNPO** | 0.213 → 0.074 (65%↓) | 0.683 | 0.565 | Good tradeoff |
| **RMU** | 0.042 → 0.022 (48%↓) | ~0.0 | **0.586** | Utility priority |
| **GradDiff** | 0.001 → 0.002 (~0%) | ~0.0 | 0.546 | Already minimal mem. |

**Key Insight**: MIA ≈ 0.5 means attacker **cannot distinguish** forget/holdout sets (optimal privacy). DPO achieves near-perfect indistinguishability.


