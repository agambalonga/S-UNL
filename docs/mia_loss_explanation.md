# Spiegazione del Calcolo della Metrica MIA_LOSS

## Introduzione

La metrica **MIA_LOSS** (Membership Inference Attack basata su Loss) è un attacco di inferenza che tenta di determinare se un dato campione faceva parte del training set del modello. Si basa sul principio che un modello tende ad avere una loss più bassa sui dati che ha visto durante il training rispetto a dati mai visti.

Nel contesto dell'**unlearning**, questa metrica viene utilizzata per verificare se il modello ha "dimenticato" efficacemente i dati del forget set: un unlearning di successo dovrebbe far sì che il modello tratti i dati del forget set in modo simile ai dati mai visti (holdout set).

## Architettura del Codice

Il calcolo della metrica MIA_LOSS coinvolge tre componenti principali:

1. **`mia_loss()`** in `__init__.py` - Entry point della metrica
2. **`mia_auc()`** in `utils.py` - Orchestratore del calcolo dell'AUC
3. **`LOSSAttack`** in `loss.py` - Implementazione specifica dell'attacco

## Flusso di Esecuzione Dettagliato

### 1. Entry Point: `mia_loss()`

```python
@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    return mia_auc(
        LOSSAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
    )
```

**Input:**
- `model`: Il modello da valutare (dopo l'unlearning)
- `data`: Dizionario con chiavi `"forget"` e `"holdout"`
  - `"forget"`: Dataset dei dati che il modello dovrebbe aver dimenticato
  - `"holdout"`: Dataset di dati mai visti dal modello
- `collators`: Funzione di collation per preparare i batch
- `batch_size`: Dimensione dei batch

**Operazione:**
Invoca la funzione generica `mia_auc()` passando la classe `LOSSAttack` come strategia di attacco.

---

### 2. Orchestratore: `mia_auc()`

```python
def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
```

Questa funzione coordina l'intero processo di calcolo dell'AUC per l'attacco MIA.

#### Step 2.1: Preparazione degli Argomenti

```python
attack_args = {
    "model": model,
    "collator": collator,
    "batch_size": batch_size,
}
attack_args.update(kwargs)
```

Prepara gli argomenti comuni per l'attacco.

#### Step 2.2: Esecuzione dell'Attacco su Entrambi i Set

```python
output = {
    "forget": attack_cls(data=data["forget"], **attack_args).attack(),
    "holdout": attack_cls(data=data["holdout"], **attack_args).attack(),
}
```

Per ogni set di dati (forget e holdout):
1. Crea un'istanza di `LOSSAttack` con i dati specifici
2. Invoca il metodo `.attack()` che restituisce:
   - `"agg_value"`: Media degli score su tutti i campioni
   - `"value_by_index"`: Dizionario `{indice: {"score": valore}}` per ogni campione

#### Step 2.3: Estrazione degli Score

```python
forget_scores = [
    elem["score"] for elem in output["forget"]["value_by_index"].values()
]
holdout_scores = [
    elem["score"] for elem in output["holdout"]["value_by_index"].values()
]
```

Estrae tutti gli score individuali per entrambi i set.

#### Step 2.4: Preparazione per il Calcolo dell'AUC

```python
scores = np.array(forget_scores + holdout_scores)
labels = np.array(
    [0] * len(forget_scores) + [1] * len(holdout_scores)
)
```

**Convenzione delle Label:**
- **Label 0**: Campioni del forget set (dovrebbero avere score più alti dopo l'unlearning)
- **Label 1**: Campioni dell'holdout set (dovrebbero avere score più bassi)

**Nota Importante:** Gli score sono **signed** (con segno) in modo tale che:
- **Score più alto** = Meno probabile che il campione sia membro del training set
- **Score più basso** = Più probabile che il campione sia membro del training set

Nel caso della loss, lo score è direttamente la loss media, quindi:
- Loss alta → Il modello "non conosce" il dato
- Loss bassa → Il modello "conosce" il dato

#### Step 2.5: Calcolo dell'AUC

```python
auc_value = roc_auc_score(labels, scores)
output["auc"], output["agg_value"] = auc_value, auc_value
```

Calcola l'AUC (Area Under the ROC Curve) usando `sklearn.metrics.roc_auc_score`.

**Interpretazione dell'AUC:**
- **AUC ≈ 1.0**: Il modello assegna loss molto più alta al forget set rispetto all'holdout → **Buon unlearning**
- **AUC ≈ 0.5**: Il modello non distingue tra forget e holdout set → **Ottimo unlearning** (il modello ha "dimenticato" completamente)
- **AUC ≈ 0.0**: Il modello assegna loss più bassa al forget set → **Pessimo unlearning** (il modello ricorda ancora i dati)

---

### 3. Implementazione dell'Attacco: `LOSSAttack`

La classe `LOSSAttack` eredita dalla classe base `Attack` e implementa l'attacco specifico basato sulla loss.

#### Classe Base: `Attack`

```python
class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        self.model = model
        self.dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
        self.setup(**kwargs)
```

**Inizializzazione:**
- Salva il modello
- Crea un `DataLoader` per iterare sui dati in batch
- Chiama `setup()` per parametri specifici dell'attacco

#### Metodo `attack()`

```python
def attack(self):
    all_scores = []
    all_indices = []

    for batch in tqdm(self.dataloader, total=len(self.dataloader)):
        indices = batch.pop("index").cpu().numpy().tolist()
        batch_values = self.compute_batch_values(batch)
        scores = [self.compute_score(values) for values in batch_values]

        all_scores.extend(scores)
        all_indices.extend(indices)

    scores_by_index = {
        str(idx): {"score": float(score)}
        for idx, score in zip(all_indices, all_scores)
    }

    return {
        "agg_value": float(np.mean(all_scores)),
        "value_by_index": scores_by_index,
    }
```

**Flusso:**
1. Itera su tutti i batch
2. Estrae gli indici dei campioni dal batch
3. Calcola i valori per il batch con `compute_batch_values()`
4. Calcola lo score per ogni campione con `compute_score()`
5. Aggrega i risultati in un dizionario

#### Implementazione in `LOSSAttack`

```python
class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        """Compute probabilities and losses for the batch."""
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        """Return the average loss for the sample."""
        return sample_stats["avg_loss"]
```

- **`compute_batch_values()`**: Delega a `evaluate_probability()` per calcolare loss e probabilità
- **`compute_score()`**: Estrae semplicemente la loss media dal dizionario delle statistiche

---

### 4. Calcolo della Loss: `evaluate_probability()`

```python
def evaluate_probability(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    avg_losses = avg_losses.cpu().numpy().tolist()
    normalized_probs = normalized_probs.cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]
```

#### Step 4.1: Forward Pass

```python
with torch.no_grad():
    output = model(**batch)
logits = output.logits
```

Esegue un forward pass del modello senza calcolare gradienti per ottenere i logits.

#### Step 4.2: Preparazione per il Calcolo della Loss

```python
labels = batch["labels"]
shifted_labels = labels[..., 1:].contiguous()
logits = logits[..., :-1, :].contiguous()
```

**Shifting per Next Token Prediction:**
- I modelli linguistici causali predicono il token successivo
- I logits alla posizione `t` predicono il token alla posizione `t+1`
- Quindi si allineano i logits con le label shiftate

**Esempio:**
```
Testo:     "Il gatto mangia"
Tokens:    [101, 234, 567, 890]
Logits:    predizioni per [234, 567, 890, <next>]
Labels:    [234, 567, 890]
```

#### Step 4.3: Calcolo della Loss Token-wise

```python
loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
```

- **`ignore_index=IGNORE_INDEX`**: Ignora i token di padding nelle label
- **`reduction="none"`**: Restituisce la loss per ogni token invece di una media
- **`.sum(dim=-1)`**: Somma la loss su tutti i token di ogni campione

#### Step 4.4: Normalizzazione per Lunghezza

```python
num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
avg_losses = losses / num_token_gt
```

Divide la loss totale per il numero di token effettivi (escludendo padding) per ottenere la **loss media per token**.

Questo è importante perché:
- Campioni più lunghi avrebbero loss totali più alte
- La normalizzazione rende i valori comparabili tra campioni di diverse lunghezze

#### Step 4.5: Calcolo della Probabilità Normalizzata

```python
normalized_probs = torch.exp(-avg_losses)
```

Converte la loss media in una probabilità normalizzata usando l'esponenziale negativo.

**Interpretazione Matematica:**

La loss media per token è:
$$\text{avg\_loss} = -\frac{1}{T}\sum_{t=1}^{T} \log P(w_t | w_{<t})$$

dove $T$ è il numero di token e $P(w_t | w_{<t})$ è la probabilità del token $w_t$ dato il contesto precedente.

Quindi:
$$\exp(-\text{avg\_loss}) = \exp\left(\frac{1}{T}\sum_{t=1}^{T} \log P(w_t | w_{<t})\right) = \left(\prod_{t=1}^{T} P(w_t | w_{<t})\right)^{1/T}$$

Questa è la **media geometrica delle probabilità** dei token, che rappresenta una misura di quanto il modello "conosca" la sequenza.

#### Step 4.6: Restituzione dei Risultati

```python
return [
    {"prob": prob, "avg_loss": avg_loss}
    for prob, avg_loss in zip(normalized_probs, avg_losses)
]
```

Restituisce una lista di dizionari, uno per ogni campione nel batch, contenente:
- `"prob"`: Probabilità normalizzata (media geometrica delle probabilità dei token)
- `"avg_loss"`: Loss media per token

---

## Schema Completo del Flusso

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. mia_loss(model, data, collators, batch_size)                │
│    - Entry point della metrica                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. mia_auc(LOSSAttack, model, data, collator, batch_size)      │
│    - Orchestrazione del calcolo                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Forget Set           │         │ Holdout Set          │
│ LOSSAttack.attack()  │         │ LOSSAttack.attack()  │
└─────────┬────────────┘         └──────────┬───────────┘
          │                                  │
          │  Per ogni batch:                 │
          │  ┌─────────────────────────┐    │
          │  │ 3. compute_batch_values │    │
          │  │    evaluate_probability │    │
          │  └────────┬────────────────┘    │
          │           │                      │
          │           ▼                      │
          │  ┌─────────────────────────┐    │
          │  │ Forward pass:           │    │
          │  │ - model(batch)          │    │
          │  │ - logits                │    │
          │  └────────┬────────────────┘    │
          │           │                      │
          │           ▼                      │
          │  ┌─────────────────────────┐    │
          │  │ Loss calculation:       │    │
          │  │ - CrossEntropyLoss      │    │
          │  │ - sum per sample        │    │
          │  └────────┬────────────────┘    │
          │           │                      │
          │           ▼                      │
          │  ┌─────────────────────────┐    │
          │  │ Normalization:          │    │
          │  │ - avg_loss = loss / T   │    │
          │  │ - prob = exp(-avg_loss) │    │
          │  └────────┬────────────────┘    │
          │           │                      │
          │           ▼                      │
          │  ┌─────────────────────────┐    │
          │  │ 4. compute_score        │    │
          │  │    return avg_loss      │    │
          │  └─────────────────────────┘    │
          │                                  │
          ▼                                  ▼
    [score₀, score₁, ...]          [score₀, score₁, ...]
          │                                  │
          └──────────────┬───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. AUC Calculation                                              │
│    - Combine scores: [forget_scores + holdout_scores]          │
│    - Create labels: [0, 0, ..., 1, 1, ...]                     │
│    - roc_auc_score(labels, scores)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   ┌──────────┐
                   │ AUC Value│
                   └──────────┘
```

---

## Formule Matematiche

### Loss Media per Token

Per un campione con $T$ token:

$$\mathcal{L}_{\text{avg}} = \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}_{\text{CE}}(w_t, \hat{w}_t)$$

dove:
- $\mathcal{L}_{\text{CE}}$ è la Cross-Entropy Loss
- $w_t$ è il token ground truth alla posizione $t$
- $\hat{w}_t$ è la distribuzione predetta dal modello

### Cross-Entropy Loss per Token

$$\mathcal{L}_{\text{CE}}(w_t, \hat{w}_t) = -\log P_{\theta}(w_t | w_{<t})$$

dove $P_{\theta}(w_t | w_{<t})$ è la probabilità assegnata dal modello al token corretto dato il contesto precedente.

### Score MIA

Per l'attacco LOSS, lo score per un campione è semplicemente:

$$\text{Score}_{\text{MIA}} = \mathcal{L}_{\text{avg}}$$

### AUC (Area Under the ROC Curve)

L'AUC misura la capacità dell'attacco di distinguere tra campioni del forget set e dell'holdout set:

$$\text{AUC} = P(\text{Score}_{\text{forget}} > \text{Score}_{\text{holdout}})$$

Nel contesto dell'unlearning:
- **AUC alta (→ 1)**: Il modello ha loss più alta sul forget set → Ha dimenticato i dati
- **AUC bassa (→ 0.5)**: Il modello non distingue tra i due set → Unlearning efficace
- **AUC molto bassa (→ 0)**: Il modello ha loss più bassa sul forget set → Non ha dimenticato

---

## Esempio Pratico

### Scenario

Supponiamo di avere:
- **Forget set**: 100 campioni di un autore specifico che il modello deve dimenticare
- **Holdout set**: 100 campioni di altri autori mai visti durante il training

### Calcolo

1. **Forget Set**:
   - Il modello calcola la loss media per ogni campione
   - Esempio di score: `[2.5, 2.8, 2.3, ..., 2.7]` (100 valori)
   - Media: 2.6

2. **Holdout Set**:
   - Il modello calcola la loss media per ogni campione
   - Esempio di score: `[3.1, 3.3, 2.9, ..., 3.2]` (100 valori)
   - Media: 3.1

3. **Labeling**:
   - Forget scores → label 0
   - Holdout scores → label 1

4. **AUC Calculation**:
   - Si costruisce la curva ROC variando la soglia di decisione
   - Si calcola l'area sotto la curva
   - In questo caso, se i forget scores sono consistentemente più bassi degli holdout scores, l'AUC sarà vicina a 0, indicando che l'attacco può facilmente identificare i campioni del forget set

### Interpretazione

- Se **AUC ≈ 0.5**: Il modello tratta il forget set e l'holdout set allo stesso modo → **Unlearning riuscito**
- Se **AUC > 0.5**: Il modello ha loss più alta sul forget set rispetto all'holdout → **Parziale dimenticamento**
- Se **AUC < 0.5**: Il modello ha loss più bassa sul forget set → **Unlearning fallito**

---

## Aspetti Chiave da Ricordare

1. **Convenzione degli Score**: Gli score sono progettati in modo che valori più alti indichino minor probabilità di membership (loss più alta = meno familiarità).

2. **Normalizzazione per Lunghezza**: La loss viene sempre normalizzata per il numero di token per rendere comparabili campioni di diverse lunghezze.

3. **Obiettivo dell'Unlearning**: Un buon unlearning dovrebbe risultare in un AUC vicino a 0.5, indicando che il modello non può più distinguere tra dati del forget set e dati mai visti.

4. **Interpretazione dell'AUC nel Contesto**: A differenza dei task di classificazione tradizionali dove AUC alta è positiva, nell'unlearning un AUC vicino a 0.5 è l'obiettivo desiderato.

5. **Efficienza Computazionale**: Il calcolo richiede solo un forward pass senza gradienti, rendendolo molto efficiente.

---

## Conclusione

La metrica MIA_LOSS fornisce una misura quantitativa dell'efficacia dell'unlearning basandosi su un principio semplice ma potente: un modello che ha veramente "dimenticato" certi dati dovrebbe trattarli come se non li avesse mai visti. L'uso dell'AUC permette di quantificare questa differenza in modo statisticamente robusto, fornendo una metrica interpretabile per valutare algoritmi di machine unlearning.
