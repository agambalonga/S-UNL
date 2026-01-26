"""Dataset handler per caricare dati augmented da JSON con controllo flessibile delle parafrasate"""

import json
import logging
from datasets import Dataset
from .qa import QADataset
from .utils import add_dataset_index

logger = logging.getLogger(__name__)


class AugmentedQADataset(QADataset):
    """
    Carica dataset augmented da file JSON con formato nidificato.

    Formato JSON atteso:
    [
        {
            "question": "Domanda originale",
            "answer": "Risposta",
            "paraphrases": ["Parafrasi 1", "Parafrasi 2", ...]
        },
        ...
    ]

    Permette di controllare quante parafrasate usare per ogni domanda originale.
    """

    def __init__(
        self,
        json_path: str,
        question_key: str = "question",
        answer_key: str = "answer",
        max_length: int = 512,
        num_paraphrases_per_question: int = 0,
        include_original: bool = True,
        tokenizer=None,
        template_args=None,
        predict_with_generate=False,
        **kwargs,
    ):
        """
        Args:
            json_path: Path al JSON con dati augmented (formato nidificato)
            question_key: Chiave per domande
            answer_key: Chiave per risposte
            max_length: Max token length
            num_paraphrases_per_question: Numero di parafrasate da usare per ogni domanda (0 = nessuna)
            include_original: Se True, include la domanda originale nel dataset
            tokenizer: Tokenizer HuggingFace
            template_args: Argomenti per il template
            predict_with_generate: Se usare generazione durante predict
        """
        # Carica JSON
        logger.info(f"Loading augmented dataset from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Espandi dataset secondo i parametri
        expanded_data = []

        for item in raw_data:
            question = item[question_key]
            answer = item[answer_key]
            paraphrases = item.get("paraphrases", [])

            # Aggiungi domanda originale
            if include_original:
                expanded_data.append(
                    {question_key: question, answer_key: answer, "type": "original"}
                )

            # Aggiungi N parafrasate
            for i in range(min(num_paraphrases_per_question, len(paraphrases))):
                expanded_data.append(
                    {
                        question_key: paraphrases[i],
                        answer_key: answer,
                        "type": f"paraphrase_{i + 1}",
                    }
                )

        logger.info(
            f"Dataset expansion: {len(raw_data)} original -> {len(expanded_data)} samples "
            f"(original={include_original}, paraphrases={num_paraphrases_per_question})"
        )

        # Converti in HF Dataset
        self.data = Dataset.from_list(expanded_data)

        # Aggiungi index (necessario per collator)
        self.data = add_dataset_index(self.data)

        logger.info(f"Loaded {len(self.data)} augmented samples")

        # Inizializza attributi (come in QADataset.__init__)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args if template_args else {}
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate
        self.fs_data = None  # No few-shot per augmented dataset

    # __len__, _process_sample e __getitem__ sono ereditati da QADataset!
    # Non serve riscriverli, funzionano già correttamente!
