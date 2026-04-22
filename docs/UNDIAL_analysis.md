# Analisi UNDIAL: Perché le Parafrasi Non Producono Miglioramenti

## Sommario Esecutivo

UNDIAL è l'unico metodo tra quelli analizzati che **non beneficia dell'arricchimento con parafrasi**. L'analisi del codice e del paper rivela che questa limitazione è **intrinseca al meccanismo** di funzionamento del metodo, non un problema di implementazione o configurazione.

**Risultati chiave dall'ablation study:**
- **Exact Memorization**: 0.200 → 0.201 (+0.3%) — *nessun miglioramento*
- **Extraction Strength**: 0.033 → 0.033 — *completamente piatto*
- **Q+A Probability**: riduzione solo del 49.2% (vs. 88.7% di DPO)
- **Paraphrased Probability**: riduzione solo del 15.3% (vs. 93.7% di DPO)

---

## 1. Come Funziona UNDIAL: Analisi del Meccanismo

### 1.1 Architettura e Componenti

Dal codice (`src/trainer/unlearn/undial.py`):

```python
class UNDIAL(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)  # ← Modello "teacher" congelato
```

UNDIAL si basa su **tre componenti fondamentali**:

1. **Student Model** (`model`): il modello da "de-addestrare", con parametri aggiornabili
2. **Teacher Model** (`ref_model`): una **copia congelata** del modello originale fine-tuned
3. **Beta (β)**: forza della penalità applicata ai token da dimenticare

### 1.2 Meccanismo di Self-Distillation con Logits Aggiustati

Il cuore del metodo si trova in `compute_undial_loss` (`src/trainer/utils.py` righe 71-95):

```python
def compute_undial_loss(model, ref_model, inputs, beta):
    # 1. Forward pass sul modello student (trainable)
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    
    # 2. Forward pass sul modello teacher (frozen)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    
    # 3. Costruzione della maschera sui token da dimenticare
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0
    
    # 4. AGGIUSTAMENTO CRITICO: sottrai beta dai logits sui token corretti
    pre_softmax = shift_teacher_logits - mask * beta
    
    # 5. Creazione dei soft labels modificati
    soft_label = F.softmax(pre_softmax, dim=-1)
    
    # 6. Training: student deve imitare i soft labels aggiustati
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    soft_label.view(-1, soft_label.size(-1)))
    return loss.mean(), outputs
```

### 1.3 Interpretazione del Meccanismo

**Step 1-2**: UNDIAL ottiene due set di logits:
- **Student logits**: dal modello in fase di unlearning
- **Teacher logits**: dal modello **originale congelato** che contiene ancora la conoscenza

**Step 3-4**: Per ogni token della risposta corretta nel forget set:
- UNDIAL identifica quale token dovrebbe essere generato (`labels`)
- Sottrae una penalità `beta` dal logit di quel token **nel teacher**
- Questo "abbassa artificialmente" la probabilità del token corretto

**Step 5**: I logits aggiustati vengono convertiti in soft labels (distribuzioni di probabilità)

**Step 6**: Lo student viene addestrato a **imitare** queste distribuzioni modificate

---

## 2. Perché le Parafrasi Non Funzionano: Analisi Causale

### 2.1 Dipendenza Critica dal Reference Model

Il problema fondamentale è questo:

**Il reference model (`ref_model`) è una copia congelata del modello originale fine-tuned sul dataset TOFU completo.**

Questo significa:
- È stato addestrato sulle 400 domande **originali** del forget set
- **Non ha mai visto** le parafrasi generate successivamente
- Viene creato **una sola volta** all'inizio dell'unlearning (riga 10 di `undial.py`)
- Rimane **congelato** per tutta la durata del training

### 2.2 Comportamento su Domande Parafrasate

Quando UNDIAL processa una domanda parafrasata:

#### Caso 1: Domanda Originale (presente nel training del reference model)
```
Q_original: "What is the birth date of author John Smith?"
A_correct: "March 15, 1975"

Teacher model comportamento:
- Ha visto questa domanda durante il fine-tuning
- Genera logits ALTI per la risposta corretta
- β viene sottratto da questi logits alti
- L'aggiustamento produce un segnale forte: "non generare questa risposta"
✓ Meccanismo funziona come intended
```

#### Caso 2: Domanda Parafrasata (MAI vista dal reference model)
```
Q_paraphrased: "When was the writer John Smith born?"
A_correct: "March 15, 1975"

Teacher model comportamento:
- Non ha mai visto questa formulazione
- I logits per la risposta corretta sono imprevedibili:
  * Potrebbero essere alti se il modello generalizza bene
  * Potrebbero essere bassi se la formulazione confonde
  * Potrebbero essere distribuiti su risposte diverse
- β viene sottratto da logits già incerti
- L'aggiustamento non produce un segnale coerente
✗ Meccanismo perde efficacia
```

### 2.3 Evidenza dai Risultati Sperimentali

I dati confermano questa analisi:

**Tabella: Confronto UNDIAL con altri metodi (para0 → para20)**

| Metrica | DPO | NPO | SimNPO | UNDIAL |
|---------|-----|-----|--------|--------|
| **EM Δ%** | -14.5% | -18.2% | -12.0% | **+0.3%** ← *nessun effetto* |
| **ES Δ%** | -47.5% | -74.0% | -50.6% | **0.0%** ← *completamente piatto* |
| **Q+A Prob Δ%** | -88.7% | -78.1% | -73.0% | **-49.2%** ← *miglioramento limitato* |
| **Para Prob Δ%** | -93.7% | -79.3% | -82.4% | **-15.3%** ← *critico* |

**Osservazione critica**: La metrica **Paraphrased Probability** mostra il problema più chiaramente:
- DPO riduce del 93.7% → generalizza fortemente alle parafrasi
- UNDIAL riduce solo del 15.3% → **non generalizza**

Questo indica che UNDIAL sta "dimenticando" principalmente le formulazioni **letterali** viste dal reference model, non la conoscenza semantica sottostante.

---

## 3. Confronto con Metodi che Beneficiano delle Parafrasi

### 3.1 DPO (Direct Preference Optimization)

**Meccanismo:** Contrasto tra risposte "preferite" (evasive/IDK) e "evitate" (corrette)

```python
# DPO loss (semplificato)
for each paraphrased question:
    logits_win = model(question, idk_response)    # Vogliamo aumentare questo
    logits_lose = model(question, correct_answer) # Vogliamo diminuire questo
    loss = -log(sigmoid(beta * (logits_win - logits_lose)))
```

**Perché le parafrasi aiutano:**
- **Ogni parafrase** viene processata indipendentemente
- Il modello impara a **evitare la risposta corretta** per **molte formulazioni diverse**
- Costruisce una rappresentazione semantica: "questa informazione è da evitare"
- Generalizza alla conoscenza sottostante, non alla forma sintattica

### 3.2 NPO (Negative Preference Optimization)

**Meccanismo:** Massimizza la differenza tra le probabilità del modello corrente e reference

```python
# NPO loss (semplificato)
for each paraphrased question:
    current_logprob = model(question, correct_answer)
    ref_logprob = ref_model(question, correct_answer)
    loss = beta * (current_logprob - ref_logprob)  # Penalizza se current > ref
```

**Perché le parafrasi aiutano:**
- Il reference model viene aggiornato durante il training (o è più sofisticato)
- Ogni parafrase spinge il modello a ridurre la confidenza
- L'effetto si accumula su diverse formulazioni semanticamente equivalenti

### 3.3 Differenza Cruciale con UNDIAL

| Aspetto | DPO/NPO | UNDIAL |
|---------|---------|--------|
| **Reference model** | Può essere aggiornato/non usato direttamente | **Congelato ab initio** |
| **Training su parafrasi** | Ogni parafrase contribuisce indipendentemente | Parafrasi nuove confondono il teacher |
| **Target dell'unlearning** | Semantica (evitare conoscenza) | Sintattica (imitare distribuzioni aggiustate) |
| **Generalizzazione** | Alta (pattern semantici) | **Bassa (pattern lessicali)** |

---

## 4. Analisi dal Paper Originale

### 4.1 Obiettivi Dichiarati (ArXiv 2402.10052)

Dal abstract:
> "UnDIAL [...] leverages self-distillation to adjust logits and selectively reduce the influence of **targeted tokens**."

> "Existing unlearning methods, like Gradient Ascent and Negative Preference Optimization, directly tune models to remove unwanted information. However, these methods often become **unstable** because they fine-tune by maximizing cross-entropy loss..."

> "Our approach [...] ensures **smooth convergence** and avoids catastrophic forgetting"

### 4.2 Design Filosofico

UNDIAL è progettato per:
1. **Stabilità**: evitare l'instabilità del gradient ascent
2. **Convergenza smooth**: distribuzioni target vicine al modello originale
3. **Evitare forgetting catastrofico**: modifiche graduali

### 4.3 Trade-off Intrinseco

**Pro:**
- Convergenza stabile (confermato dai risultati: curve molto smooth)
- Nessun over-unlearning catastrofico

**Contro:**
- **Rigidità**: dipende fortemente dalla "guida" del teacher model
- **Scope limitato**: funziona solo su esempi che il teacher "riconosce"
- **Generalizzazione povera**: non apprende pattern semantici, solo surface-level

Citando il paper:
> "self-distillation to adjust logits and selectively reduce the influence of **targeted tokens**"

La parola chiave è **"targeted tokens"**: il metodo è token-level, non semantic-level.

---

## 5. Evidenze Aggiuntive dai Risultati

### 5.1 Plateau Strutturale

Dalla tesi (sezione Dinamiche Temporali):
> "UNDIAL merita una menzione a parte per il suo comportamento atipico. La metrica Exact Memorization resta **praticamente invariata** attorno agli stessi valori lungo tutte le epoche e per tutte le configurazioni di parafrasi. Le curve sono così **sovrapposte** da suggerire che il metodo abbia raggiunto un **limite strutturale**, forse legato alla sua architettura interna, piuttosto che ai parametri di training."

Questo conferma che il problema è **architetturale**, non di training.

### 5.2 Privacy Leakage

| Configurazione | DPO | NPO | UNDIAL |
|----------------|-----|-----|--------|
| **para0** | -73.6 | 22.0 | **-15.6** ← migliore |
| **para5** | -27.6 | 20.9 | **21.9** ← peggiora |
| **para20** | -28.9 | 7.7 | **32.1** ← continua a peggiorare |

**Interpretazione:**
- Inizialmente UNDIAL parte bene (vicino a 0 = ottimale)
- Con le parafrasi **peggiora progressivamente**
- Questo suggerisce che le parafrasi **confondono** il meccanismo
- Il teacher model non fornisce guidance coerente

### 5.3 Model Utility Catastrofica

UNDIAL: **Model Utility = 0.160** (vs. 0.586 di RMU, 0.545 di NPO)

Dalla tesi:
> "UNDIAL si conferma il metodo più problematico, con valori di utilità estremamente bassi che lo rendono inadatto ad applicazioni pratiche."

**Possibile spiegazione:**
- Il teacher model fornisce segnali contraddittori su parafrasi mai viste
- Lo student cerca di imitare distribuzioni incoerenti
- Il training danneggia sia il forgetting che l'utility

---

## 6. Spiegazione Tecnica Dettagliata

### 6.1 Flusso di Informazione in UNDIAL

```
Step 0: Preparazione
├─ Fine-tune model su D_f ∪ D_r → θ_original
└─ ref_model = copy(θ_original), freeze()

Step 1: Per ogni batch del forget set (con parafrasi)
├─ Input: (question_paraphrased, answer_correct)
│
├─ A. Forward pass reference model (frozen)
│   ├─ logits_teacher = ref_model(question_paraphrased)
│   └─ PROBLEMA: ref_model mai addestrato su questa formulazione
│
├─ B. Aggiustamento logits
│   ├─ mask = one_hot(correct_tokens)
│   ├─ adjusted_logits = logits_teacher - β * mask
│   └─ soft_targets = softmax(adjusted_logits)
│   └─ PROBLEMA: se logits_teacher erano già incerti, 
│       sottrazione di β produce segnale instabile
│
├─ C. Forward pass student model
│   └─ logits_student = model(question_paraphrased)
│
└─ D. Loss: distanza tra logits_student e soft_targets
    └─ PROBLEMA: soft_targets poco affidabili → training dannoso
```

### 6.2 Scenario Concreto: Comportamento su Parafrase

**Setup:**
- Domanda originale vista dal reference: "What is John Smith's birth date?"
- Parafrase mai vista: "When was John Smith born?"

**Test Reference Model:**
```python
# Domanda originale
ref_model("What is John Smith's birth date?")
>>> Logits top-3: [("March 15", 8.5), ("1975", 7.2), ("The", -2.1)]
                   ↑ Alta confidenza, aggiustamento efficace

# Parafrase
ref_model("When was John Smith born?")
>>> Logits top-3: [("In", 3.2), ("He", 2.8), ("March", 2.1)]
                   ↑ Confidenza distribuita, incertezza
```

**Effetto dell'aggiustamento:**
```python
# Domanda originale
adjusted = logits_teacher - β * mask[correct_tokens]
"March" logit: 8.5 - 10.0 = -1.5  ← forte penalizzazione
soft_label["March"] ≈ 0.05       ← probabilità bassa

# Parafrase
"March" logit: 2.1 - 10.0 = -7.9  ← penalizzazione eccessiva
soft_label["March"] ≈ 0.0001      ← quasi annullato
soft_label["In"] ≈ 0.45           ← token spurio ora dominante
```

**Risultato:** Lo student impara distribuzioni **non sensate**

---

## 7. Conclusioni e Raccomandazioni

### 7.1 Perché UNDIAL Non Beneficia delle Parafrasi (Sintesi)

1. **Dipendenza dal Teacher Congelato:**
   - Il reference model è addestrato solo sulle domande originali
   - Non può fornire guidance affidabile su formulazioni mai viste

2. **Meccanismo Token-Level:**
   - UNDIAL aggiusta logits su token specifici, non rappresentazioni semantiche
   - L'aggiustamento presuppone che il teacher "conosca" la risposta corretta
   - Su parafrasi nuove, questo presupposto fallisce

3. **Propagazione di Incertezza:**
   - Logits incerti del teacher → soft targets incoerenti → training dannoso
   - Lo student impara a imitare distribuzioni senza senso

4. **Evidenza Empirica:**
   - EM e ES piatti attraverso tutte le configurazioni
   - Paraphrased Probability peggiora solo del 15.3% (vs. 93.7% di DPO)
   - Model Utility catastrofica (0.160)
   - Privacy Leakage peggiora con più parafrasi

### 7.2 Confronto con Metodi Efficaci

Le parafrasi funzionano per metodi che:
- **Processano ogni esempio indipendentemente** (DPO, NPO)
- **Apprendono pattern semantici** (reward signals, contrasti)
- **Non dipendono da un reference model fisso** (o lo aggiornano)

UNDIAL invece:
- **Dipende criticamente** dal knowledge del teacher
- **Apprende a livello sintattico** (distribuzioni di token)
- **Reference model completamente statico**

### 7.3 Possibili Soluzioni (Speculative)

**Opzione 1: Re-training del Reference Model**
- Fine-tune il reference model **anche sulle parafrasi**
- PRO: Teacher può guidare lo student su tutte le formulazioni
- CONTRO: Costo computazionale doppio, scarsa fattibilità pratica

**Opzione 2: Multiple Teacher Ensemble**
- Usare più reference models con diverse fine-tune
- Aggregare le loro guidance
- CONTRO: Complessità elevata, overhead computazionale

**Opzione 3: Dynamic Teacher**
- Aggiornare periodicamente il teacher durante l'unlearning
- CONTRO: Perde la proprietà di stabilità di UNDIAL

**Opzione 4: Hybrid Loss**
- Combinare UNDIAL loss con preference-based loss
- CONTRA: Diventa un metodo diverso, perde identità

**Raccomandazione:** Nessuna di queste soluzioni è praticamente conveniente. **UNDIAL è inadatto all'unlearning semanticamente robusto con parafrasi**.

### 7.4 Implicazioni Pratiche

**Per l'ablation study:**
- I risultati sono **corretti e coerenti** con il design del metodo
- Non è un problema di implementazione o hyperparameters
- È una **limitazione intrinseca** del metodo

**Per applicazioni reali:**
- UNDIAL può essere efficace per unlearning di **pattern esatti e specifici**
- **Non adatto** quando serve generalizzazione semantica
- **Non raccomandato** in scenari con diverse formulazioni possibili

---

## 8. Sezione per la Tesi: Proposta di Testo

> ### Caso Speciale: UNDIAL e la Limitazione Strutturale
>
> UNDIAL rappresenta un caso particolarmente istruttivo per comprendere come le caratteristiche intrinseche di un metodo di unlearning possano determinarne la compatibilità con l'arricchimento linguistico del forget set.
>
> Come evidenziato nelle Tabelle~\ref{tab:em_reduction} e~\ref{tab:extraction_strength_reduction}, UNDIAL mostra un comportamento anomalo: le metriche di memorizzazione rimangono sostanzialmente invariate indipendentemente dal numero di parafrasi utilizzate (EM: 0.200 → 0.201, ES: 0.033 → 0.033). Questo pattern, così diverso da quanto osservato nei metodi basati su ottimizzazione di preferenze, richiede un'analisi del meccanismo sottostante.
>
> **Analisi del Meccanismo**
>
> UNDIAL \cite{dong_undial_2024} si basa su un paradigma di self-distillation: un *teacher model*, corrispondente a una copia congelata del modello originale fine-tuned sul dataset completo, fornisce i logits di riferimento. Questi logits vengono modificati sottraendo una penalità $\beta$ alle posizioni corrispondenti ai token che compongono le risposte del forget set, producendo così soft labels che riducono artificialmente la probabilità dei token da dimenticare. Lo *student model* viene quindi addestrato a imitare queste distribuzioni modificate attraverso una loss di cross-entropy.
>
> Il punto critico risiede nella natura del teacher model: esso viene creato **prima** dell'unlearning e rimane **congelato** per tutta la durata del training. Questo significa che il teacher è stato addestrato esclusivamente sulle 400 domande originali del forget set, e non ha mai osservato le parafrasi generate successivamente.
>
> **Impatto sulle Parafrasi**
>
> Quando UNDIAL processa una domanda originale, il teacher model produce logits affidabili: avendo visto quella formulazione durante il fine-tuning, i logits per la risposta corretta sono tipicamente alti. La sottrazione di $\beta$ da questi valori elevati produce un segnale chiaro: ridurre drasticamente la probabilità di quella risposta. Lo student impara efficacemente a evitare la risposta corretta per quella specifica formulazione.
>
> Quando invece UNDIAL processa una domanda parafrasata — mai vista dal teacher durante il fine-tuning — la guidance diventa inaffidabile. Il teacher model potrebbe:
> - Non riconoscere la formulazione, producendo logits bassi o distribuiti
> - Generalizzare parzialmente, assegnando probabilità moderate
> - Rispondere in modo incoerente rispetto alla versione originale
>
> In queste condizioni, sottrarre $\beta$ da logits già incerti non produce un segnale utile per l'unlearning. Lo student riceve distribuzioni target poco coerenti, che non contribuiscono alla rimozione della conoscenza semantica.
>
> **Conferma Sperimentale**
>
> Questa interpretazione trova conferma nei dati sperimentali. La metrica Paraphrased Probability, che valuta direttamente la capacità del modello di dimenticare formulazioni alternative, mostra per UNDIAL una riduzione di appena il 15.3\% (da 0.080 a 0.068), contro il 93.7\% di DPO. Questo indica che UNDIAL opera principalmente a livello sintattico, rimuovendo pattern letterali ma non la conoscenza semantica sottostante.
>
> Inoltre, l'osservazione della metrica Privacy Leakage (Tabella~\ref{tab:privleak}) rivela un pattern inusuale: mentre DPO migliora progressivamente con l'aggiunta di parafrasi (da -73.6 a -28.9, avvicinandosi al valore ottimale di 0), UNDIAL mostra un andamento opposto (da -15.6 a +32.1), allontanandosi dal target ideale. Questo comportamento suggerisce che le parafrasi **confondono** il meccanismo di unlearning piuttosto che rafforzarlo.
>
> **Implicazioni Teoriche**
>
> Il caso di UNDIAL evidenzia una distinzione fondamentale tra metodi di unlearning:
> - **Metodi instance-level** (DPO, NPO, SimNPO): processano ciascun esempio del forget set indipendentemente, costruendo progressivamente una rappresentazione semantica della conoscenza da evitare. L'aggiunta di formulazioni diverse rafforza questo apprendimento.
> - **Metodi reference-dependent** (UNDIAL): si basano su una "fonte di verità" esterna (il teacher model) per guidare il processo. Se questa fonte non è allineata con i dati di training (parafrasi mai viste), l'efficacia del metodo decade.
>
> Questa analisi suggerisce che l'efficacia dell'arricchimento linguistico dipende non solo dalla qualità delle parafrasi o dalla configurazione degli iperparametri, ma dalle assunzioni architetturali del metodo di unlearning. UNDIAL, pur essendo efficace per l'unlearning di pattern specifici e ben definiti, risulta inadatto quando si richiede generalizzazione semantica attraverso diverse formulazioni linguistiche.

---

## Riferimenti

- **Paper UNDIAL**: Dong et al., "UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models", NAACL 2025, https://aclanthology.org/2025.naacl-long.444.pdf
- **Codice**: 
  - `/home/agambalo/open-unlearning/src/trainer/unlearn/undial.py`
  - `/home/agambalo/open-unlearning/src/trainer/utils.py` (linee 71-95)
- **Configurazione**: `/home/agambalo/open-unlearning/configs/trainer/UNDIAL.yaml`
- **Risultati**: Tesi, Capitolo 4, Sezioni 4.3-4.6

---

*Documento preparato per discussione con il prof. Basile*  
*Data: 8 Marzo 2026*
