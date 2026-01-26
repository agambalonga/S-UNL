"""
Dataset handler per DPO con augmented questions (parafrasate) + IDK responses.
Combina la logica di AugmentedQADataset e QAwithIdkDataset.
"""

import json
import logging
import torch
from datasets import Dataset
from .qa import QADataset
from .utils import add_dataset_index

logger = logging.getLogger(__name__)


class AugmentedQAwithIdkDataset(QADataset):
    """
    Dataset per DPO che:
    1. Carica domande augmented da JSON (originali + parafrasate)
    2. Usa risposte IDK come alternate response
    3. Ritorna struttura {'original': ..., 'alternate': ...} per DPO
    """
    
    def __init__(
        self,
        json_path: str,
        idk_path: str,
        return_original: bool = True,
        question_key: str = "question",
        answer_key: str = "answer",
        num_paraphrases_per_question: int = 0,
        include_original: bool = True,
        filter_types: list = None,
        tokenizer=None,
        template_args=None,
        max_length: int = 512,
        predict_with_generate: bool = False,
        **kwargs
    ):
        """
        Args:
            json_path: Path al JSON con dati augmented
            idk_path: Path al file JSONL con risposte IDK
            return_original: Se True, ritorna struttura DPO con original/alternate
            question_key: Chiave per domande
            answer_key: Chiave per risposte originali
            num_paraphrases_per_question: Numero di parafrasate da usare per ogni domanda (0 = nessuna)
            include_original: Se True, include la domanda originale nel dataset
            filter_types: Lista di tipi da includere (None = tutti, deprecato se usi num_paraphrases)
            tokenizer: Tokenizer HuggingFace
            template_args: Template arguments per chat formatting
            max_length: Max token length
            predict_with_generate: Se True, prepara per generazione
        """
        self.json_path = json_path
        self.idk_path = idk_path
        self.return_original = return_original
        self.filter_types = filter_types
        
        # Carica risposte IDK
        logger.info(f"Loading IDK responses from {idk_path}")
        with open(idk_path, 'r', encoding='utf-8') as f:
            self.idk_responses = f.readlines()
        logger.info(f"Loaded {len(self.idk_responses)} IDK responses")
        
        # Carica JSON augmented (formato nidificato)
        logger.info(f"Loading augmented dataset from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Espandi dataset secondo parametri (come in AugmentedQADataset)
        expanded_data = []
        
        for item in raw_data:
            question = item[question_key]
            answer = item[answer_key]
            paraphrases = item.get("paraphrases", [])
            
            # Aggiungi domanda originale
            if include_original:
                expanded_data.append({
                    question_key: question,
                    answer_key: answer,
                    "type": "original"
                })
            
            # Aggiungi N parafrasate
            for i in range(min(num_paraphrases_per_question, len(paraphrases))):
                expanded_data.append({
                    question_key: paraphrases[i],
                    answer_key: answer,
                    "type": f"paraphrase_{i+1}"
                })
        
        logger.info(
            f"Dataset expansion for DPO: {len(raw_data)} original -> {len(expanded_data)} samples "
            f"(original={include_original}, paraphrases={num_paraphrases_per_question})"
        )
        
        # Filtra per tipo se richiesto (retrocompatibilità)
        if filter_types:
            original_len = len(expanded_data)
            expanded_data = [item for item in expanded_data if item.get("type") in filter_types]
            logger.info(
                f"Filtered {original_len} -> {len(expanded_data)} samples "
                f"(types: {filter_types})"
            )
        
        # Converti in HF Dataset
        self.data = Dataset.from_list(expanded_data)
        self.data = add_dataset_index(self.data)
        
        logger.info(f"Loaded {len(self.data)} augmented samples for DPO with IDK")
        
        # Inizializza attributi per parent class
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args if template_args else {}
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate
        self.fs_data = None
    
    def item_with_idk(self, question, index):
        """
        Genera item con risposta IDK random.
        Stessa logica di QAwithIdkDataset.item_with_idk()
        """
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(
            question=question, 
            answer=idk_response,
            index=index
        )
        return idk_item
    
    def __getitem__(self, idx):
        """
        Ritorna struttura compatibile con DPO:
        {
            'original': {input_ids, labels, attention_mask, index},
            'alternate': {input_ids, labels, attention_mask, index}
        }
        """
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        
        # Process original answer (risposta vera)
        original_item = self._process_sample(
            question=question,
            answer=answer,
            index=index
        )
        
        # Process alternate answer (IDK random)
        alternate_item = self.item_with_idk(question, index)
        
        return_item = {
            "original": original_item,
            "alternate": alternate_item
        }
        
        return return_item if self.return_original else return_item["alternate"]