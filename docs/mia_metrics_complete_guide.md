# Guida Completa alle Metriche MIA (Membership Inference Attacks)

## Indice

1. [Introduzione](#introduzione)
2. [Architettura Comune](#architettura-comune)
3. [MIA LOSS](#mia-loss)
4. [MIA Min-K](#mia-min-k)
5. [MIA Min-K++](#mia-min-k-plus-plus)
6. [MIA GradNorm](#mia-gradnorm)
7. [MIA ZLIB](#mia-zlib)
8. [MIA Reference](#mia-reference)
9. [Confronto tra le Metriche](#confronto-tra-le-metriche)
10. [Interpretazione dell'AUC nell'Unlearning](#interpretazione-dellauc-nellunlearning)

---

## Introduzione

Le **Membership Inference Attacks (MIA)** sono tecniche che tentano di determinare se un dato campione faceva parte del training set di un modello. Nel contesto dell'**unlearning**, queste metriche valutano quanto efficacemente un modello ha "dimenticato" i dati del forget set.

### Principio Generale

Un modello tende a comportarsi diversamente su:
- **Dati visti durante il training**: Loss bassa, alta confidenza, gradienti maggiori
- **Dati mai visti**: Loss alta, bassa confidenza, gradienti minori

Un **unlearning efficace** dovrebbe far sì che il modello tratti i dati del forget set come se non li avesse mai visti (simili all'holdout set).

### Convenzioni Importanti

1. **Score Signed**: Tutti gli score MIA seguono la convenzione: **score più alto = meno probabile membership**
2. **AUC Interpretation**: 
   - AUC → 1.0: Forget set riconoscibile (pessimo unlearning)
   - AUC → 0.5: Forget e holdout indistinguibili (ottimo unlearning)
   - AUC → 0.0: Holdout più riconoscibile del forget (problematico)

---

## Architettura Comune

Tutte le metriche MIA condividono la stessa architettura basata sulla classe `Attack`:

```python
class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        self.model = model
        self.dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """Override per configurare parametri specifici"""
        pass
    
    def compute_batch_values(self, batch):
        """Override per calcolare statistiche dal batch"""
        raise NotImplementedError
    
    def compute_score(self, sample_stats):
        """Override per calcolare lo score MIA da statistiche del campione"""
        raise NotImplementedError
    
    def attack(self):
        """Esegue l'attacco completo"""
        all_scores = []
        all_indices = []
        
        for batch in tqdm(self.dataloader):
            indices = batch.pop("index").cpu().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]
            
            all_scores.extend(scores)
            all_indices.extend(indices)
        
        return {
            "agg_value": float(np.mean(all_scores)),
            "value_by_index": {str(idx): {"score": float(score)} 
                              for idx, score in zip(all_indices, all_scores)}
        }
```

### Funzione `mia_auc`

Tutte le metriche utilizzano questa funzione per calcolare l'AUC:

```python
def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    # Esegue l'attacco su forget e holdout
    output = {
        "forget": attack_cls(data=data["forget"], **attack_args).attack(),
        "holdout": attack_cls(data=data["holdout"], **attack_args).attack(),
    }
    
    # Estrae gli score
    forget_scores = [elem["score"] for elem in output["forget"]["value_by_index"].values()]
    holdout_scores = [elem["score"] for elem in output["holdout"]["value_by_index"].values()]
    
    # Calcola AUC (label 0=forget, 1=holdout)
    scores = np.array(forget_scores + holdout_scores)
    labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
    auc_value = roc_auc_score(labels, scores)
    
    return output
```

---

## MIA LOSS

**Paper**: [Membership Inference Attacks Against Machine Learning Models (2017)](https://ieeexplore.ieee.org/abstract/document/8429311)

### Descrizione

L'attacco più semplice e diretto: utilizza la **loss media per token** come score di membership. Un modello tende ad avere loss più bassa su dati che ha visto durante il training.

### Implementazione

```python
class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        return evaluate_probability(self.model, batch)
    
    def compute_score(self, sample_stats):
        return sample_stats["avg_loss"]
```

### Calcolo Dettagliato

#### 1. Forward Pass
```python
output = model(**batch)
logits = output.logits  # Shape: (batch_size, seq_len, vocab_size)
```

#### 2. Allineamento Logits-Labels
```python
labels = batch["labels"]
shifted_labels = labels[..., 1:].contiguous()  # Target tokens
logits = logits[..., :-1, :].contiguous()       # Predictions
```

#### 3. Loss per Token
```python
loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
```

#### 4. Normalizzazione
```python
num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
avg_losses = losses / num_token_gt
```

### Formula Matematica

Per un campione con $T$ token:

$$\text{Score}_{\text{LOSS}} = \mathcal{L}_{\text{avg}} = \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}_{\text{CE}}(w_t, \hat{w}_t)$$

dove:
$$\mathcal{L}_{\text{CE}}(w_t, \hat{w}_t) = -\log P_{\theta}(w_t | w_{<t})$$

### Vantaggi e Svantaggi

**Vantaggi:**
- Semplice e intuitivo
- Computazionalmente efficiente (solo forward pass)
- Baseline solida per confronti

**Svantaggi:**
- Può essere influenzato dalla lunghezza del testo
- Non considera la distribuzione dei token rari vs comuni
- Sensibile al dominio dei dati

---

## MIA Min-K

**Paper**: [Min-K% Prob: A Simple and Effective Method for MIA (2023)](https://arxiv.org/pdf/2310.16789.pdf)

### Descrizione

Invece di usare la media su tutti i token, **Min-K** si concentra sui token con le **k% probabilità più basse**. L'intuizione è che i dati di training contengono token "memorizzati" facilmente, mentre altri potrebbero essere difficili; i token con probabilità più bassa sono più informativi per l'inferenza di membership.

### Implementazione

```python
class MinKProbAttack(Attack):
    def setup(self, k=0.2, **kwargs):
        self.k = k  # Percentuale (es. 0.2 = 20%)
    
    def compute_batch_values(self, batch):
        return tokenwise_logprobs(self.model, batch, grad=False)
    
    def compute_score(self, sample_stats):
        lp = sample_stats.cpu().numpy()  # Log probabilities per token
        if lp.size == 0:
            return 0
        
        num_k = max(1, int(len(lp) * self.k))
        sorted_vals = np.sort(lp)  # Ordina in ordine crescente
        return -np.mean(sorted_vals[:num_k])  # Media dei k% più bassi (negativi)
```

### Funzione `tokenwise_logprobs`

```python
def tokenwise_logprobs(model, batch, grad=False):
    output = model(**batch)
    logits = output.logits  # (bsz, seq_len, vocab_size)
    
    # Log softmax per ottenere log probabilità
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    
    # Estrae le log-prob dei token target
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    
    # Filtra solo i token con label (esclude padding)
    log_probs_batch = []
    for i in range(batch_size):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero()[0][:-1]
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        log_probs_batch.append(target_log_probs[i, start_idx-1:end_idx])
    
    return log_probs_batch
```

### Formula Matematica

Per un campione con log-probabilità $\{\ell_1, \ell_2, \ldots, \ell_T\}$ ordinati: $\ell_{(1)} \leq \ell_{(2)} \leq \cdots \leq \ell_{(T)}$:

$$\text{Score}_{\text{Min-K}} = -\frac{1}{K} \sum_{i=1}^{K} \ell_{(i)}$$

dove $K = \lfloor k \cdot T \rfloor$ e $k$ è la percentuale (tipicamente 0.2).

### Intuizione

- **Token facili** (alta probabilità): Il modello li predice bene sia su training che test data
- **Token difficili** (bassa probabilità): Più informativi per distinguere membership
  - Su training data: Il modello potrebbe averli "memorizzati" parzialmente
  - Su test data: Probabilità consistentemente bassa

Concentrandosi sui token con probabilità più bassa, Min-K ottiene un segnale più forte per la membership inference.

### Vantaggi e Svantaggi

**Vantaggi:**
- Più robusto della media semplice
- Migliore performance empirica su modelli LLM
- Riduce l'influenza dei token "facili" che mascherano il segnale

**Svantaggi:**
- Richiede tuning del parametro k
- Può essere sensibile a outlier
- Computazionalmente simile a LOSS

---

## MIA Min-K++

**Paper**: Estensione di Min-K con normalizzazione basata sulla distribuzione del vocabolario

### Descrizione

**Min-K++** migliora Min-K normalizzando le log-probabilità usando la **media e varianza della distribuzione sull'intero vocabolario**. Questo rende gli score comparabili tra diversi token, indipendentemente dalla loro difficoltà intrinseca.

### Implementazione

```python
class MinKPlusPlusAttack(MinKProbAttack):
    def compute_batch_values(self, batch):
        vocab_log_probs = tokenwise_vocab_logprobs(self.model, batch, grad=False)
        token_log_probs = tokenwise_logprobs(self.model, batch, grad=False)
        return [
            {"vocab_log_probs": vlp, "token_log_probs": tlp}
            for vlp, tlp in zip(vocab_log_probs, token_log_probs)
        ]
    
    def compute_score(self, sample_stats):
        all_probs = sample_stats["vocab_log_probs"]  # Shape: (T, V)
        target_prob = sample_stats["token_log_probs"]  # Shape: (T,)
        
        if len(target_prob) == 0:
            return 0
        
        # Calcola media e varianza sulla distribuzione del vocabolario
        mu = (torch.exp(all_probs) * all_probs).sum(-1)  # E[log P]
        sigma = (torch.exp(all_probs) * torch.square(all_probs)).sum(-1) - torch.square(mu)
        
        # Normalizzazione z-score
        sigma = torch.clamp(sigma, min=1e-6)
        scores = (target_prob.cpu().numpy() - mu.cpu().numpy()) / torch.sqrt(sigma).cpu().numpy()
        
        # Prendi i k% più bassi
        num_k = max(1, int(len(scores) * self.k))
        return -np.mean(sorted(scores)[:num_k])
```

### Funzione `tokenwise_vocab_logprobs`

```python
def tokenwise_vocab_logprobs(model, batch, grad=False):
    output = model(**batch)
    log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)[:, :-1, :]
    # Shape: (batch_size, seq_len-1, vocab_size)
    
    log_probs_batch = []
    for i in range(batch_size):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero()[0][:-1]
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        # Restituisce la distribuzione completa per ogni token
        log_probs_batch.append(log_probs[i, start_idx-1:end_idx])  # Shape: (N, V)
    
    return log_probs_batch
```

### Formula Matematica

Per ogni token $t$, calcola lo z-score normalizzato:

$$\mu_t = \mathbb{E}_{w \sim P_t}[\log P_t(w)] = \sum_{w \in \mathcal{V}} P_t(w) \cdot \log P_t(w)$$

$$\sigma_t^2 = \mathbb{E}_{w \sim P_t}[(\log P_t(w))^2] - \mu_t^2$$

$$z_t = \frac{\log P_t(w_t^*) - \mu_t}{\sigma_t}$$

dove $w_t^*$ è il token ground truth alla posizione $t$.

Lo score finale:

$$\text{Score}_{\text{Min-K++}} = -\frac{1}{K} \sum_{i=1}^{K} z_{(i)}$$

dove $z_{(1)} \leq z_{(2)} \leq \cdots \leq z_{(T)}$ sono gli z-score ordinati.

### Intuizione

La normalizzazione basata sulla distribuzione del vocabolario compensa per:
- **Token comuni** (es. "the", "and"): Hanno naturalmente alta probabilità
- **Token rari** (es. nomi propri, termini tecnici): Hanno naturalmente bassa probabilità

Lo z-score misura quanto la probabilità del token target si discosta dalla distribuzione attesa, rendendo gli score comparabili indipendentemente dalla frequenza del token.

### Vantaggi e Svantaggi

**Vantaggi:**
- Più robusto ai token rari
- Normalizzazione automatica per difficoltà intrinseca
- Performance superiore su dataset eterogenei

**Svantaggi:**
- Computazionalmente più costoso (richiede distribuzione completa)
- Complessità maggiore nell'implementazione
- Richiede più memoria (salva distribuzioni vocab-size)

---

## MIA GradNorm

**Paper**: [Scalable Extraction of Training Data from (Production) Language Models (2024)](https://arxiv.org/abs/2402.17012)

### Descrizione

**GradNorm** utilizza la **norma dei gradienti** rispetto ai parametri del modello come indicatore di membership. L'intuizione è che i dati di training generano gradienti con pattern diversi rispetto ai dati nuovi, poiché il modello è stato ottimizzato su di essi.

### Implementazione

```python
class GradNormAttack(Attack):
    def setup(self, p, **kwargs):
        if p not in [1, 2, float("inf")]:
            raise ValueError(f"Invalid p-norm value: {p}")
        self.p = p  # Norma L1, L2, o L∞
    
    def compute_batch_values(self, batch):
        self.model.train()  # Abilita gradienti
        
        # Calcola log-prob per ogni campione
        batch_log_probs = tokenwise_logprobs(self.model, batch, grad=True)
        batch_loss = [-torch.mean(lps) for lps in batch_log_probs]
        
        batch_grad_norms = []
        for sample_loss in batch_loss:
            sample_grad_norms = []
            self.model.zero_grad()
            sample_loss.backward()  # Backprop per singolo campione
            
            # Calcola norma gradiente per ogni parametro
            for param in self.model.parameters():
                if param.grad is not None:
                    sample_grad_norms.append(param.grad.detach().norm(p=self.p))
            
            batch_grad_norms.append(torch.stack(sample_grad_norms).mean())
        
        self.model.eval()
        return batch_grad_norms
    
    def compute_score(self, sample_stats):
        return sample_stats.cpu().to(torch.float32).numpy()
```

### Formula Matematica

Per un campione $x$ con loss $\mathcal{L}(x; \theta)$, lo score è:

$$\text{Score}_{\text{GradNorm}} = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \left\| \frac{\partial \mathcal{L}(x; \theta)}{\partial \theta} \right\|_p$$

dove:
- $\Theta$ sono i parametri del modello
- $\|\cdot\|_p$ è la norma $L_p$ (tipicamente $p=2$)

### Intuizione

**Perché i gradienti sono informativi?**

1. **Training Data**: Durante il training, i parametri sono stati aggiornati per minimizzare la loss su questi dati. I gradienti su training data tendono ad avere:
   - Magnitude più alta (il modello è ancora "sensibile" a questi dati)
   - Pattern specifici correlati all'ottimizzazione

2. **New Data**: Su dati mai visti:
   - Gradienti con magnitude diversa (tipicamente più bassa o con pattern diverso)
   - Meno correlazione con la geometria del loss landscape appresa

### Nota Importante

⚠️ **Warning**: GradNorm azzera i gradienti del modello durante il calcolo. Non usare questo attacco in modalità che interferirebbe con gradienti accumulati durante il training.

### Vantaggi e Svantaggi

**Vantaggi:**
- Accesso a informazioni di secondo ordine (curvatura del loss landscape)
- Performance eccellente su modelli grandi
- Cattura memorizzazione più profonda dei semplici score di loss

**Svantaggi:**
- **Molto costoso computazionalmente** (richiede backward pass per ogni campione)
- Richiede molta memoria (gradienti per tutti i parametri)
- Interferisce con il training (azzera gradienti)
- Non scalabile a batch grandi

---

## MIA ZLIB

**Paper**: [Extracting Training Data from Large Language Models (2021)](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf)

### Descrizione

**ZLIB** normalizza la loss del modello usando l'**entropia di compressione ZLIB** del testo. L'idea è che la loss dovrebbe essere contestualizzata rispetto alla complessità intrinseca del testo: un testo più comprimibile (più predicibile) dovrebbe naturalmente avere loss più bassa.

### Implementazione

```python
class ZLIBAttack(Attack):
    def setup(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer or self.model.tokenizer
    
    def compute_batch_values(self, batch):
        # Calcola loss
        eval_results = evaluate_probability(self.model, batch)
        
        # Estrae il testo decodificato
        texts = extract_target_texts_from_processed_data(self.tokenizer, batch)
        
        return [{"loss": r["avg_loss"], "text": t} 
                for r, t in zip(eval_results, texts)]
    
    def compute_score(self, sample_stats):
        text = sample_stats["text"]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        return sample_stats["loss"] / zlib_entropy
```

### Funzione `extract_target_texts_from_processed_data`

```python
def extract_target_texts_from_processed_data(tokenizer, batch):
    labels = batch["labels"]
    # Filtra token ignorati (padding)
    labels = [elem[elem != IGNORE_INDEX] for elem in labels]
    # Decodifica
    texts = [tokenizer.decode(elem.tolist(), skip_special_tokens=True) 
             for elem in labels]
    return texts
```

### Formula Matematica

$$\text{Score}_{\text{ZLIB}} = \frac{\mathcal{L}_{\text{avg}}}{|\text{zlib.compress}(x)|}$$

dove:
- $\mathcal{L}_{\text{avg}}$ è la loss media per token
- $|\text{zlib.compress}(x)|$ è la lunghezza in bytes del testo compresso con ZLIB

### Intuizione

**Perché normalizzare con ZLIB?**

1. **Testi ripetitivi/prevedibili**: 
   - ZLIB compression alta (bytes compressi bassi)
   - Loss naturalmente bassa anche su dati nuovi
   - Score ZLIB = loss / compression → relativizza la loss

2. **Testi casuali/complessi**:
   - ZLIB compression bassa (bytes compressi alti)
   - Loss naturalmente alta
   - Score ZLIB = loss / compression → normalizza l'aspettativa

La compressione ZLIB approssima la complessità di Kolmogorov del testo, fornendo una baseline di quanto sia "predicibile" il testo indipendentemente dal modello.

### Esempio

```python
# Testo ripetitivo
text1 = "hello hello hello hello hello"
loss1 = 1.5
zlib1 = 20  # Molto comprimibile
score1 = 1.5 / 20 = 0.075

# Testo casuale
text2 = "qxzp klmn wvty rstu"
loss2 = 3.0
zlib2 = 25  # Meno comprimibile
score2 = 3.0 / 25 = 0.12

# Il testo ripetitivo ha score più basso anche se la loss assoluta è diversa
```

### Vantaggi e Svantaggi

**Vantaggi:**
- Compensa per complessità intrinseca del testo
- Utile su dataset con variabilità di complessità
- Economico computazionalmente (compressione è veloce)

**Svantaggi:**
- ZLIB non cattura perfettamente la complessità semantica
- Dipende dal tokenizer (encoding del testo)
- Performance limitata su testi brevi
- Meno interpretabile dei metodi basati solo su loss

---

## MIA Reference

### Descrizione

**Reference-based MIA** confronta la loss del modello target (dopo unlearning) con quella di un **modello di riferimento** (solitamente il modello originale prima dell'unlearning, o un modello generico). La differenza di loss è usata come score.

### Implementazione

```python
class ReferenceAttack(Attack):
    def setup(self, reference_model, **kwargs):
        self.reference_model = reference_model
    
    def compute_batch_values(self, batch):
        # Loss su modello di riferimento
        ref_results = evaluate_probability(self.reference_model, batch)
        
        # Loss su modello target
        target_results = evaluate_probability(self.model, batch)
        
        return [
            {"target_loss": t["avg_loss"], "ref_loss": r["avg_loss"]}
            for t, r in zip(target_results, ref_results)
        ]
    
    def compute_score(self, sample_stats):
        return sample_stats["target_loss"] - sample_stats["ref_loss"]
```

### Entry Point

```python
@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        kwargs["reference_model_path"],
        torch_dtype=model.dtype,
        device_map={"": model.device},
    )
    
    return mia_auc(
        ReferenceAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        reference_model=reference_model,
    )
```

### Formula Matematica

$$\text{Score}_{\text{Reference}} = \mathcal{L}_{\text{target}}(x) - \mathcal{L}_{\text{ref}}(x)$$

dove:
- $\mathcal{L}_{\text{target}}(x)$ è la loss del modello dopo unlearning
- $\mathcal{L}_{\text{ref}}(x)$ è la loss del modello di riferimento

### Intuizione

**Interpretazione dello Score:**

1. **Score > 0** (target_loss > ref_loss):
   - Il modello target ha loss più alta del riferimento
   - Il modello target "ha dimenticato" il dato
   - Indica unlearning efficace sul campione

2. **Score ≈ 0** (target_loss ≈ ref_loss):
   - Entrambi i modelli trattano il dato similmente
   - Comportamento coerente tra modelli

3. **Score < 0** (target_loss < ref_loss):
   - Il modello target ha loss più bassa del riferimento
   - Il modello target "conosce meglio" il dato
   - Può indicare overfitting o unlearning inefficace

### Scelta del Modello di Riferimento

**Opzioni comuni:**

1. **Modello Pre-unlearning**: Il modello originale prima dell'unlearning
   - Pro: Mostra esattamente l'effetto dell'unlearning
   - Contro: Richiede salvare il modello originale

2. **Modello Generico**: Un modello pre-trained mai finetuned sui dati
   - Pro: Rappresenta conoscenza "naturale" del dominio
   - Contro: Può non catturare il task specifico

3. **Modello su Retain Set**: Modello trainato solo sui dati da mantenere
   - Pro: Baseline ideale teorica
   - Contro: Costoso (richiede re-training)

### Vantaggi e Svantaggi

**Vantaggi:**
- Contestualizza la loss rispetto a un baseline significativo
- Isola l'effetto dell'unlearning dalla difficoltà intrinseca
- Interpretabile: differenza diretta di loss

**Svantaggi:**
- Richiede un modello di riferimento (memoria raddoppiata)
- Due forward pass per ogni campione (2x computazione)
- Dipende fortemente dalla scelta del modello di riferimento
- Può essere instabile se i modelli divergono troppo

---

## Confronto tra le Metriche

### Tabella Comparativa

| Metrica | Computazione | Memoria | Interpretabilità | Performance | Iperparametri |
|---------|--------------|---------|------------------|-------------|---------------|
| **LOSS** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Nessuno |
| **Min-K** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | k |
| **Min-K++** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | k |
| **GradNorm** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | p |
| **ZLIB** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Nessuno |
| **Reference** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Modello ref |

### Complessità Computazionale

Per $N$ campioni, batch size $B$, sequenza length $T$, vocab size $V$, parametri $|\Theta|$:

| Metrica | Forward Pass | Backward Pass | Complessità Totale |
|---------|--------------|---------------|-------------------|
| LOSS | $N$ | 0 | $O(N \cdot T \cdot V)$ |
| Min-K | $N$ | 0 | $O(N \cdot T \cdot V)$ |
| Min-K++ | $N$ | 0 | $O(N \cdot T \cdot V)$ (ma salva $T \times V$) |
| GradNorm | $N$ | $N$ | $O(N \cdot T \cdot V \cdot |\Theta|)$ |
| ZLIB | $N$ | 0 | $O(N \cdot T \cdot V + N \cdot T_{\text{compress}})$ |
| Reference | $2N$ | 0 | $O(2N \cdot T \cdot V)$ |

### Quando Usare Quale Metrica

#### LOSS
**Usa quando:**
- Vuoi una baseline veloce
- Risorse computazionali limitate
- Interpretabilità è prioritaria
- Dataset omogeneo

**Non usare quando:**
- Dataset con testi di lunghezze molto variabili
- Presenza di token rari che dominano il segnale

#### Min-K
**Usa quando:**
- Vuoi migliorare su LOSS senza costo aggiuntivo significativo
- Dataset con mix di token facili/difficili
- Hai tempo per tuning di k

**Non usare quando:**
- Sequenze molto corte (pochi token per k%)
- Non puoi fare grid search su k

#### Min-K++
**Usa quando:**
- Hai memoria sufficiente per distribuzioni vocab
- Dataset con alta variabilità di token (rari + comuni)
- Performance è prioritaria

**Non usare quando:**
- Modello molto grande (vocab size enorme)
- Risorse di memoria limitate

#### GradNorm
**Usa quando:**
- Vuoi la massima performance
- Hai GPU potenti e tempo
- Dataset piccolo/medio
- Cerchi memorizzazione profonda

**Non usare quando:**
- Risorse limitate
- Dataset grande (milioni di campioni)
- Valutazione frequente richiesta

#### ZLIB
**Usa quando:**
- Dataset con variabilità di complessità testuale
- Testi con ripetizioni o pattern
- Vuoi normalizzare per complessità intrinseca

**Non usare quando:**
- Testi molto brevi
- Dataset già normalizzato/preprocessato pesantemente

#### Reference
**Usa quando:**
- Hai accesso a modello di riferimento appropriato
- Vuoi isolare effetto dell'unlearning
- Memoria sufficiente per due modelli

**Non usare quando:**
- Nessun modello di riferimento appropriato disponibile
- Memoria limitata
- Incertezza su quale riferimento usare

---

## Interpretazione dell'AUC nell'Unlearning

### Formula dell'AUC

$$\text{AUC} = P(\text{Score}_{\text{forget}} > \text{Score}_{\text{holdout}})$$

### Scale di Interpretazione

| AUC Range | Interpretazione | Qualità Unlearning | Azione |
|-----------|-----------------|-------------------|---------|
| **0.95 - 1.00** | Forget set chiaramente riconoscibile | ❌ Pessimo | Re-train con parametri diversi |
| **0.75 - 0.95** | Forte distinzione tra forget e holdout | ⚠️ Insufficiente | Aumenta strength unlearning |
| **0.60 - 0.75** | Moderata distinzione | ⚠️ Parziale | Fine-tune parametri |
| **0.45 - 0.60** | Debole distinzione | ✅ Buono | Monitora altre metriche |
| **0.40 - 0.45** | Quasi indistinguibili | ✅✅ Ottimo | Success! |
| **0.00 - 0.40** | Holdout più riconoscibile | ⚠️ Anomalo | Verifica overfitting |

### Curva ROC Tipica

```
TPR ▲ 
    │     Perfect Unlearning (AUC=0.5)
1.0 │    ╱│╲
    │   ╱ │ ╲
    │  ╱  │  ╲
    │ ╱   │   ╲_____ Bad Unlearning (AUC→1.0)
0.5 │╱    │    
    │     │
    │     │
0.0 └─────┴──────────────► FPR
    0    0.5            1.0
```

### Multi-Metric Evaluation

L'unlearning efficace dovrebbe mostrare:

```
✅ MIA AUC ≈ 0.5    (Indistinguibilità)
✅ Retain Accuracy alta (Conoscenza mantenuta)
✅ Forget Accuracy bassa (Conoscenza rimossa)
✅ Model Utility preservata (Performance generale)
```

**Red Flags:**
```
🚩 MIA AUC > 0.8 ma Forget Accuracy bassa
   → Il modello risponde male ma è ancora riconoscibile
   → Possibile mode collapse

🚩 MIA AUC ≈ 0.5 ma Retain Accuracy bassa
   → L'unlearning ha danneggiato troppo il modello
   → Over-unlearning

🚩 MIA AUC ≈ 0.5 ma Forget Accuracy alta
   → Il modello ricorda ancora i fatti
   → Unlearning superficiale
```

---

## Best Practices

### 1. Usa Multiple Metriche

Non affidarti mai a una sola metrica MIA. Consigliato:
```
- MIA LOSS (baseline)
- MIA Min-K (robusto)
- MIA Reference (se disponibile)
```

### 2. Report Completi

Report sempre:
- AUC value
- AUC confidence interval (bootstrap)
- Score distributions (forget vs holdout)
- Sample size per set

### 3. Analisi per Sottogruppi

Valuta MIA separatamente su:
- Testi lunghi vs corti
- Token comuni vs rari
- Diversi domini/topic (se applicabile)

### 4. Confronto con Baseline

Includi sempre:
- Modello originale (pre-unlearning)
- Modello retrained (da scratch su retain set)
- Modello random init (worst case)

### 5. Threshold Tuning

Se usi MIA per decision-making:
- Scegli threshold basato su trade-off desiderato
- Valida su hold-out set separato
- Considera costi asimmetrici (false positives vs negatives)

---

## Conclusione

Le metriche MIA forniscono strumenti potenti per valutare l'unlearning:

1. **LOSS**: Baseline semplice e interpretabile
2. **Min-K**: Miglioramento robusto focalizzato su token informativi
3. **Min-K++**: Versione normalizzata per dataset eterogenei
4. **GradNorm**: Massima performance con costo computazionale alto
5. **ZLIB**: Normalizzazione per complessità intrinseca
6. **Reference**: Contestualizzazione rispetto a baseline esplicito

La scelta dipende da:
- **Risorse disponibili** (computazione, memoria)
- **Tipo di dataset** (lunghezza, complessità, eterogeneità)
- **Trade-off precision-cost**
- **Interpretabilità richiesta**

Un evaluation completo dovrebbe usare **almeno 2-3 metriche** diverse per catturare aspetti complementari del membership leakage.
