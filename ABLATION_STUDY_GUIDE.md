# 📊 Ablation Study - Documentazione

Questa documentazione spiega come eseguire l'ablation study sull'unlearning con parafrasi e come generare i grafici dei risultati.

---

## 🎯 Obiettivo

Verificare l'efficacia di diversi metodi di unlearning variando:
1. **Numero di parafrasi**: 0, 5, 10, 15, 20 (epoca finale) - 0 = baseline senza augmentation
2. **Numero di epoche**: 0, 5, 10, 15, 20 (training progression)

Per 6 metodi di unlearning: **GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL**

---

## 📁 File Principali

### 1. `scripts/paraphrases_ablation.sh`
Script bash che esegue tutti gli esperimenti di unlearning variando il numero di parafrasi.

### 2. `scripts/plot_ablation_results.py`
Script Python che genera grafici e tabelle dai risultati.

### 3. `configs/experiment/unlearn/tofu/paraphrased.yaml`
Configurazione per l'unlearning con dataset augmented.

### 4. `configs/experiment/unlearn/tofu/idk_paraphrased.yaml`
Configurazione specifica per DPO con dataset augmented.

---

## 🚀 Come Lanciare l'Ablation Study

### Passo 1: Preparazione

Assicurati di avere:
- ✅ Dataset augmented: `data/tofu_forget10_augmented.json` (con 20 parafrasi)
- ✅ Modello base: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- ✅ GPU con almeno 16GB VRAM
- ✅ Python environment configurato

### Passo 2: Lancio Training

```bash
cd /home/agambalo/open-unlearning

# Esegui l'ablation study (tutti i metodi con 0, 5, 10, 15, 20 parafrasi)
bash scripts/paraphrases_ablation.sh
```


### Passo 3: Monitoraggio

Durante l'esecuzione, puoi monitorare:

```bash
# Logs del training corrente
tail -f saves/unlearn/tofu_*/training.log

# Uso GPU
watch -n 1 nvidia-smi

# Checkpoint generati
ls -lh saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_*/checkpoint-*/
```

### Passo 4: Generazione Grafici

Dopo che il training è completato:

```bash
# Genera tutti i grafici (parafrasi + epoche)
python scripts/plot_ablation_results.py

# Oppure specifica epoche/parafrasi custom
python scripts/plot_ablation_results.py --paraphrases 5 10 15 20 --epochs 5 10 15 20

# Output in cartella custom
python scripts/plot_ablation_results.py --output-dir plots/my_ablation
```

---

## 📊 Output Generati

### Struttura Cartelle Training

```
saves/unlearn/
├── tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_para0_ep20/
│   └── checkpoint-*/evals/TOFU_SUMMARY.json
├── tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_para5_ep20/
│   ├── checkpoint-5/evals/TOFU_SUMMARY.json
│   ├── checkpoint-10/evals/TOFU_SUMMARY.json
│   ├── checkpoint-15/evals/TOFU_SUMMARY.json
│   └── checkpoint-20/evals/TOFU_SUMMARY.json    ← Epoca finale
├── tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_para10_ep20/
│   └── checkpoint-*/evals/TOFU_SUMMARY.json
├── tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_para15_ep20/
├── tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_para20_ep20/
├── tofu_Llama-3.2-1B-Instruct_forget10_NPO_para5_ep20/
├── ... (24 cartelle totali)
```

### Grafici Generati

**Cartella output**: `plots/ablation_study/`

#### A) Grafici vs Parafrasi (Epoca Finale) - 5 file

```
memorization_score_vs_paraphrases.png    # Linee: metodi vs parafrasi
privacy_score_vs_paraphrases.png         # Linee: metodi vs parafrasi  
utility_score_vs_paraphrases.png         # Linee: metodi vs parafrasi
aggregate_score_vs_paraphrases.png       # Linee: metodi vs parafrasi
all_metrics_vs_paraphrases_grid.png      # Grid 2×2 con tutte le metriche
```

**Esempio grafico**:
```
Memorization Score vs Number of Paraphrases (Final Epoch)
    ↑
0.9 |        GradDiff ●━━●━━●━━●━━●
0.8 |    NPO ●━━●━━●━━●━━●
0.7 |  SimNPO ●━━●━━●━━●━━●
    |____________________________
        0   5   10  15  20  → Paraphrases
```

#### B) Grafici vs Epoche (Per Config Parafrasi) - 20 file

Per **ogni** configurazione di parafrasi (0, 5, 10, 15, 20):

```
# Con 0 parafrasi (baseline):
memorization_score_vs_epochs_para0.png
privacy_score_vs_epochs_para0.png
utility_score_vs_epochs_para0.png
aggregate_score_vs_epochs_para0.png
all_metrics_vs_epochs_para0_grid.png

# Con 5 parafrasi:
memorization_score_vs_epochs_para5.png
privacy_score_vs_epochs_para5.png
utility_score_vs_epochs_para5.png
aggregate_score_vs_epochs_para5.png
all_metrics_vs_epochs_para5_grid.png

# Con 10 parafrasi:
*_vs_epochs_para10.png (×5)

# Con 15 parafrasi:
*_vs_epochs_para15.png (×5)

# Con 20 parafrasi:
*_vs_epochs_para20.png (×5)
```

**Esempio grafico**:
```
Memorization Score vs Epochs (10 Paraphrases)
    ↑
0.9 |        GradDiff ●━━●━━●━━●
0.8 |    NPO ●━━●━━●━━●
0.7 |  SimNPO ●━━●━━●━━●
    |________________________
        5   10  15  20  → Epochs
```

#### C) Tabelle CSV - 5 file

```
results_by_paraphrases.csv       # Risultati finali per parafrasi
results_by_epochs_para5.csv      # Progressione epoche (5 parafrasi)
results_by_epochs_para10.csv     # Progressione epoche (10 parafrasi)
results_by_epochs_para15.csv     # Progressione epoche (15 parafrasi)
results_by_epochs_para20.csv     # Progressione epoche (20 parafrasi)
```

**Esempio CSV**:
```csv
trainer,num_paraphrases,aggregate_score,memorization_score,privacy_score,utility_score
GradDiff,0,0.7845,0.8923,0.4312,0.7012
GradDiff,5,0.8234,0.9145,0.4523,0.7321
GradDiff,10,0.8567,0.9289,0.4712,0.7654
NPO,5,0.7891,0.8745,0.4234,0.7123
...
```

**Totale file generati**: ~35 visualizzazioni

---

## 📈 Metriche Analizzate

### 1. **Memorization Score** (↑ meglio)
Formula: `HM(1-ES, 1-EM, 1-Para.Prob, 1-Truth Ratio)`
- Misura quanto bene il modello ha "dimenticato"
- Valori alti = buon unlearning

### 2. **Privacy Score** (↑ meglio)
Formula: `HM(sLOSS, sZLib, sMin-k, sMink++)`
- Misura la privacy tramite MIA attacks
- Valori alti = più simile al retain model (buon unlearning)

### 3. **Utility Score** (↑ meglio)
Formula: `HM(Model Utility, Fluency)`
- Misura quanto il modello mantiene utilità generale
- Valori alti = buona performance su retain data

### 4. **Aggregate Score** (↑ meglio)
Formula: `HM(Memorization, Utility)` 
- Score complessivo secondo TOFU paper (Table 6)
- Bilancia dimenticanza e utilità

---

## 🔍 Come Interpretare i Risultati

### Domande a cui Rispondere

1. **Quale metodo funziona meglio?**
   - Guarda `aggregate_score_vs_paraphrases.png`
   - Metodo con linea più alta = migliore

2. **Più parafrasi = migliori risultati?**
   - Confronta para0 (baseline) con para5/10/15/20
   - Se linee salgono da 0 → le parafrasi aiutano
   - Se linee scendono dopo un picco → troppe parafrasi possono degradare
   - Para0 mostra performance senza data augmentation

3. **Convergenza durante training?**
   - Guarda `*_vs_epochs_para*.png`
   - Se linee si stabilizzano → convergenza raggiunta
   - Se continuano a salire → potrebbero servire più epoche

4. **Trade-off memorization vs utility?**
   - Confronta `memorization_score` con `utility_score`
   - Metodo ideale: alto in entrambe

### Esempio Analisi

```
GradDiff: Memorization=0.92, Utility=0.75 → Forte unlearning, utility ok
NPO:      Memorization=0.85, Utility=0.88 → Unlearning buono, utility alta
DPO:      Memorization=0.78, Utility=0.91 → Unlearning moderato, utility ottima

→ Scelta dipende da priorità: dimenticare tutto (GradDiff) vs mantenere utilità (DPO)
```

---

## ⚙️ Parametri Configurabili

### Nel Training (`paraphrases_ablation.sh`)

```bash
# Numero di epoche totali
num_epochs=20  # Cambia per training più lungo/corto

# Configurazioni parafrasi da testare
paraphrase_counts=(0 5 10 15 20)  # 0 = baseline senza parafrasi

# Batch size (per gestire OOM)
per_device_train_batch_size=2
gradient_accumulation_steps=4
```

### Nei Grafici (`plot_ablation_results.py`)

```bash
# Epoche da analizzare
python plot_ablation_results.py --epochs 5 10 15 20

# Parafrasi da analizzare  
python plot_ablation_results.py --paraphrases 0 5 10 15 20

# Directory input/output
python plot_ablation_results.py \
    --base-dir saves/unlearn \
    --output-dir plots/my_results
```

---

## 🐛 Troubleshooting

### OOM durante training

**Problema**: `CUDA out of memory`

**Soluzioni**:
```bash
# 1. Riduci batch size
per_device_train_batch_size=1
gradient_accumulation_steps=8

# 2. Usa gradient checkpointing (già attivo)
model.gradient_checkpointing=true

# 3. Riduci numero di parafrasi per run
paraphrase_counts=(5 10)  # Invece di (5 10 15 20)
```

### Grafici vuoti

**Problema**: `No results found`

**Soluzioni**:
- Verifica che il training sia completato
- Controlla che esistano i file `TOFU_SUMMARY.json`
- Verifica pattern nel codice: 
  ```bash
  ls saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_*/checkpoint-*/evals/
  ```

### Training interrotto

**Problema**: Script fermato a metà

**Soluzioni**:
- Modifica lo script per saltare esperimenti già completati
- Controlla l'exit code dell'ultimo comando
- Riavvia dal punto di interruzione modificando i loop

---

## 📝 Note Importanti

1. **Spazio disco**: Ogni esperimento occupa ~8GB
   - 30 esperimenti × 8GB = **~240GB totali**

2. **Checkpoints intermedi**: Salvati ad ogni epoca
   - Utili per analisi progressione
   - Eliminabili dopo generazione grafici per risparmiare spazio

3. **Riproducibilità**: I seed sono fissi nei config
   - Risultati dovrebbero essere consistenti tra run

4. **GPU Requirements**: 
   - Minimo: 16GB VRAM (RTX 5000 Ada ✅)
   - Raccomandato: 24GB per batch size più grandi

---

## 📚 Output per il Report

Dopo aver generato tutti i grafici, avrai:

### Per il report scritto:
- `results_by_paraphrases.csv` → Tabella finale
- `all_metrics_vs_paraphrases_grid.png` → Grafico principale

### Per analisi dettagliata:
- Grafici individuali per ogni metrica
- Tabelle CSV per ogni configurazione
- Progressione temporale (epoche) per ogni setup

### Conclusioni da trarre:
1. Quale metodo è migliore overall?
2. Le parafrasi migliorano rispetto al baseline (0)?
3. Quante parafrasi sono ottimali?
4. Trade-off tra memorization e utility
5. Convergenza: servono davvero 20 epoche?

---

## 🎓 Riferimenti

- **TOFU Benchmark**: https://arxiv.org/abs/2401.06121
- **Metriche Aggregate**: Vedi Table 6 nel paper
- **Dataset**: locuslab/TOFU su HuggingFace

---

Buona fortuna con gli esperimenti! 🚀
