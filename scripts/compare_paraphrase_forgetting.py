#!/usr/bin/env python3
"""
Script per confrontare le risposte di due modelli su domande parafrasate.

Interroga due modelli con domande parafrasate del forget set e salva i risultati in JSON.

Usage:
    python scripts/compare_paraphrase_forgetting.py \
        --baseline-path <path_to_baseline> \
        --enriched-path <path_to_enriched> \
        --output <output_file.json> \
        --num-samples 200
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


class ModelComparator:
    """Confronta le risposte di due modelli."""
    
    def __init__(
        self,
        baseline_path: str,
        enriched_path: str,
        device: str = "cuda"
    ):
        self.device = device
        
        print(f"🔄 Caricamento baseline model: {baseline_path}")
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.baseline_model.eval()
        
        print(f"🔄 Caricamento enriched model: {enriched_path}")
        self.enriched_model = AutoModelForCausalLM.from_pretrained(
            enriched_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.enriched_model.eval()
        
        print(f"🔄 Caricamento tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(baseline_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_paraphrased_data(self, forget_split: str = "forget10") -> List[Dict]:
        """Carica domande parafrasate dal file JSON."""
        paraphrase_file = f"data/tofu_{forget_split}_augmented.json"
        
        if not os.path.exists(paraphrase_file):
            print(f"❌ File parafrasi non trovato: {paraphrase_file}")
            print("   Genera con: python scripts/generation/generate_paraphrases.py")
            sys.exit(1)
        
        with open(paraphrase_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Caricati {len(data)} esempi con parafrasi")
        return data
    
    def format_prompt(self, question: str) -> str:
        """Formatta la domanda come prompt per il modello."""
        messages = [
            {"role": "user", "content": question}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_response(
        self,
        model: AutoModelForCausalLM,
        prompt: str,
        max_new_tokens: int = 150
    ) -> str:
        """Genera una risposta dal modello."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Estrai solo la parte generata (senza il prompt)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response_text.strip()
    
    def compare_on_sample(
        self,
        original_question: str,
        paraphrased_question: str,
        true_answer: str,
        paraphrase_idx: int = 0
    ) -> Dict:
        """Confronta i due modelli su una singola domanda."""
        prompt = self.format_prompt(paraphrased_question)
        
        # Genera risposte
        baseline_response = self.generate_response(self.baseline_model, prompt)
        enriched_response = self.generate_response(self.enriched_model, prompt)
        
        return {
            "original_question": original_question,
            "paraphrased_question": paraphrased_question,
            "true_answer": true_answer,
            "paraphrase_idx": paraphrase_idx,
            "baseline_response": baseline_response,
            "enriched_response": enriched_response
        }
    



def main():
    parser = argparse.ArgumentParser(
        description="Confronta le risposte di due modelli su domande parafrasate"
    )
    
    parser.add_argument(
        "--baseline-path",
        type=str,
        required=True,
        help="Path al modello baseline"
    )
    parser.add_argument(
        "--enriched-path",
        type=str,
        required=True,
        help="Path al modello enriched"
    )
    parser.add_argument(
        "--forget-split",
        type=str,
        default="forget10",
        help="Split TOFU da usare (default: forget10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Numero di domande da testare (default: 10)"
    )
    parser.add_argument(
        "--paraphrase-idx",
        type=int,
        default=0,
        help="Quale parafrasi usare per ogni domanda, 0-19 (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File di output dove scrivere i risultati"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed per sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Crea directory di output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("🔬 CONFRONTO RISPOSTE MODELLI SU PARAFRASI")
    print("="*80)
    print(f"Baseline: {args.baseline_path}")
    print(f"Enriched: {args.enriched_path}")
    print(f"Numero campioni: {args.num_samples}")
    print(f"Indice parafrasi: {args.paraphrase_idx}")
    print(f"Split: {args.forget_split}")
    print("="*80)
    
    # Inizializza comparator
    comparator = ModelComparator(
        baseline_path=args.baseline_path,
        enriched_path=args.enriched_path
    )
    
    # Carica dati parafrasati
    paraphrased_data = comparator.load_paraphrased_data(args.forget_split)
    
    # Sampling casuale
    np.random.seed(args.seed)
    sample_indices = np.random.choice(
        len(paraphrased_data),
        size=min(args.num_samples, len(paraphrased_data)),
        replace=False
    )
    
    # Confronta su ogni sample
    results = []
    
    for i, idx in enumerate(tqdm(sample_indices, desc="Generazione risposte")):
        sample = paraphrased_data[idx]
        
        # Usa parafrasi se specificato, altrimenti domanda originale
        original_question = sample['question']
        if args.paraphrase_idx >= 0 and args.paraphrase_idx < len(sample['paraphrases']):
            paraphrased_question = sample['paraphrases'][args.paraphrase_idx]
        else:
            paraphrased_question = sample['question']
        
        result = comparator.compare_on_sample(
            original_question=original_question,
            paraphrased_question=paraphrased_question,
            true_answer=sample['answer'],
            paraphrase_idx=args.paraphrase_idx
        )
        
        results.append(result)
    
    # Prepara output JSON
    output_data = {
        "metadata": {
            "baseline_path": args.baseline_path,
            "enriched_path": args.enriched_path,
            "num_samples": len(results),
            "paraphrase_idx": args.paraphrase_idx,
            "forget_split": args.forget_split,
            "seed": args.seed
        },
        "samples": results
    }
    
    # Salva JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"✅ Completato! Analizzati {len(results)} campioni")
    print(f"💾 Risultati salvati in: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
