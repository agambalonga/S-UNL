"""
Script OFFLINE per generare parafrasate usando Llama-3.2-3B-Instruct
Eseguire PRIMA del training, non a runtime.

Usage:
    python scripts/generate_paraphrases.py \
        --split forget10 \
        --output data/tofu_forget10_augmented.json \
        --num-paraphrases 2
"""

import os
import json
import argparse
from typing import List
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaParaphraseGenerator:
    """Genera parafrasate usando Llama-3.2-3B-Instruct"""

    SYSTEM_PROMPT = """You are a paraphrasing expert. Your ONLY job is to rephrase questions using different words while keeping EXACTLY the same meaning.

CRITICAL RULE: Every detail (names, dates, places, numbers) MUST appear in the paraphrase.

Examples:

Original: "What is the profession of John Smith's father?"
Paraphrase 1: What job does John Smith's father have?
Paraphrase 2: What does John Smith's father do for a living?
Paraphrase 3: What is the occupation of John Smith's father?

Original: "What city was Maria Garcia born in on 12/05/1990?"
Paraphrase 1: In which city was Maria Garcia born on 12/05/1990?
Paraphrase 2: Where was Maria Garcia born on 12/05/1990?
Paraphrase 3: What is Maria Garcia's birthplace on 12/05/1990?

Original: "How has Peter's background in medicine influenced his work?"
Paraphrase 1: In what way has Peter's medical background affected his work?
Paraphrase 2: How has Peter's experience in medicine shaped his work?
Paraphrase 3: What influence has Peter's medical background had on his work?

Now paraphrase this question {num} times. Output ONLY the paraphrases, one per line:"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        num_paraphrases: int = 2,
        device: str = "cuda",
    ):
        self.num_paraphrases = num_paraphrases
        self.device = device

        print(f"📥 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ottimizzazione: usa bfloat16 senza quantizzazione per velocità
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True,  # Abilita KV caching per velocità
        )
        self.model.eval()
        print("✅ Model loaded!")

    def generate_paraphrases(self, question: str) -> List[str]:
        """Genera parafrasate per una singola domanda - OTTIMIZZATO"""

        # Prompt semplicissimo per modelli piccoli
        user_message = f"{question}"

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.format(num=self.num_paraphrases),
            },
            {"role": "user", "content": user_message},
        ]

        # Applica chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Genera TUTTE le parafrasi in una sola chiamata (molto più veloce!)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Più token per 20 parafrasi
                temperature=0.5,  # BASSA temperatura per fedeltà
                do_sample=True,
                top_p=0.85,  # Più conservativo
                repetition_penalty=1.1,  # Leggera penalità
                use_cache=True,  # KV caching per velocità
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Parse output
        paraphrases = self._parse_output(generated)

        # Se non sono abbastanza, usa fallback
        if len(paraphrases) < self.num_paraphrases:
            fallback = self._simple_paraphrases(
                question, self.num_paraphrases - len(paraphrases)
            )
            paraphrases.extend(fallback)

        return paraphrases[: self.num_paraphrases]

    def _parse_output(self, text: str) -> List[str]:
        """Estrae parafrasate dall'output del modello - SOLO DOMANDE VALIDE"""
        import re

        lines = text.strip().split("\n")
        paraphrases = []

        for line in lines:
            # Rimuovi numerazione e prefissi comuni
            cleaned = line.strip()
            cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", cleaned)
            cleaned = re.sub(r"^[•*\-]\s*", "", cleaned)

            # Rimuovi prefissi testuali
            for prefix in [
                "Question:",
                "Paraphrase:",
                "Answer:",
                "Original:",
                "Rephrased:",
            ]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()

            # Rimuovi punti interrogativi multipli (?, ??, ???, etc.)
            cleaned = re.sub(r"\?+", "?", cleaned)

            # Rimuovi caratteri strani alla fine
            cleaned = re.sub(r"\?\.\?$", "?", cleaned)
            cleaned = re.sub(r"[\.;,]+\?$", "?", cleaned)

            # Verifica che sia una DOMANDA valida
            if len(cleaned) > 10:
                # Se non finisce con '?', prova ad aggiungerlo
                if not cleaned.endswith("?"):
                    # Verifica se inizia con parole interrogative
                    question_starters = [
                        "what",
                        "who",
                        "where",
                        "when",
                        "why",
                        "how",
                        "can",
                        "could",
                        "would",
                        "should",
                        "do",
                        "does",
                        "did",
                        "is",
                        "are",
                        "was",
                        "were",
                        "may",
                        "might",
                        "has",
                        "have",
                        "had",
                        "will",
                        "shall",
                    ]
                    if any(cleaned.lower().startswith(q) for q in question_starters):
                        cleaned = cleaned + "?"
                    else:
                        # Salta se non sembra una domanda
                        continue

                # Verifica che non ci siano caratteri strani
                if cleaned.count("?") == 1 and cleaned.endswith("?"):
                    paraphrases.append(cleaned)

        return paraphrases

    def _simple_paraphrases(self, question: str, num_needed: int) -> List[str]:
        """Fallback con template semplici per garantire il numero richiesto"""
        templates = [
            lambda q: f"Can you tell me {q.lower()}",
            lambda q: f"Do you know {q.lower()}",
            lambda q: f"Could you explain {q.lower()}",
            lambda q: f"I'd like to know {q.lower()}",
            lambda q: f"Please tell me {q.lower()}",
            lambda q: f"What about {q.lower()}",
            lambda q: f"Can you explain {q.lower()}",
            lambda q: f"Would you tell me {q.lower()}",
            lambda q: f"I want to know {q.lower()}",
            lambda q: f"Could you tell me {q.lower()}",
            lambda q: f"May I know {q.lower()}",
            lambda q: f"Can I ask {q.lower()}",
            lambda q: f"Tell me {q.lower()}",
            lambda q: f"Explain {q.lower()}",
            lambda q: f"Describe {q.lower()}",
            lambda q: f"Share with me {q.lower()}",
            lambda q: f"Let me know {q.lower()}",
            lambda q: f"I need to know {q.lower()}",
            lambda q: f"Help me understand {q.lower()}",
            lambda q: f"Clarify {q.lower()}",
        ]

        fallbacks = []
        for template in templates:
            if len(fallbacks) >= num_needed:
                break
            try:
                fallbacks.append(template(question))
            except Exception:
                continue

        return fallbacks[:num_needed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locuslab/TOFU")
    parser.add_argument("--split", default="forget10")
    parser.add_argument("--output", default="data/tofu_forget10_augmented.json")
    parser.add_argument("--num-paraphrases", type=int, default=2)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    # Carica dataset originale
    print(f"📥 Loading {args.dataset}/{args.split}...")
    dataset = load_dataset(args.dataset, args.split, split="train")
    print(f"✅ Loaded {len(dataset)} samples")

    # Inizializza generator
    generator = LlamaParaphraseGenerator(
        model_name=args.model,
        num_paraphrases=args.num_paraphrases,
    )

    # Genera dataset augmented con struttura nidificata
    augmented_data = []

    for item in tqdm(dataset, desc="Generating paraphrases"):
        question = item["question"]
        answer = item["answer"]

        # Genera parafrasate
        paraphrases = []
        try:
            paraphrases = generator.generate_paraphrases(question)
            # print all paraphrases
            print(f"\nOriginal Question: {question}")
            for idx, para in enumerate(paraphrases):
                print(f"Paraphrase {idx + 1}: {para}")
        except Exception as e:
            print(f"\n⚠️ Error generating paraphrases: {e}")
            # Continua con lista vuota se fallisce

        # Salva in formato nidificato
        augmented_data.append(
            {
                "question": question,
                "answer": answer,
                "paraphrases": paraphrases,  # Lista di N parafrasate
            }
        )

    # Salva
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to {args.output}")
    print(f"   Original questions: {len(dataset)}")
    print(f"   Paraphrases per question: {args.num_paraphrases}")
    print(f"   Total entries: {len(augmented_data)}")

    # Statistiche parafrasate generate
    total_paraphrases = sum(len(item["paraphrases"]) for item in augmented_data)
    avg_paraphrases = total_paraphrases / len(augmented_data) if augmented_data else 0
    print(f"   Avg paraphrases generated: {avg_paraphrases:.2f}")


if __name__ == "__main__":
    main()
