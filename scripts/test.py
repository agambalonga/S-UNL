from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import os

# Usa percorso assoluto
model_path = "/home/agambalo/Universita/open-unlearning/saves/unlearn/DPO_EXPERIMENT/"

# Verifica che il percorso esista
if not os.path.exists(model_path):
    print(f"❌ Errore: Percorso non trovato: {model_path}")
    print("📂 Percorsi disponibili:")
    base_path = "/home/agambalo/Universita/open-unlearning/saves/unlearn/"
    if os.path.exists(base_path):
        for folder in os.listdir(base_path):
            print(f"   - {folder}")
    exit(1)

print("🚀 Caricamento modello...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

print("📚 Caricamento dataset TOFU forget10...")
forget_data = load_dataset("locuslab/TOFU", "forget10_perturbed")
print(type(forget_data))  # Verifica il tipo di forget_data
train_dataset = forget_data["train"]  # ← Questo è il Dataset con 400 righe
print(f"✅ Dataset caricato: {len(train_dataset)} esempi")

print("📚 Caricamento dataset TOFU retain90...")
retain_data = load_dataset("locuslab/TOFU", "retain90")
retain_dataset = retain_data["train"]
print(f"✅ Retain dataset caricato: {len(retain_dataset)} esempi")

print("🧪 TEST CON DOMANDE FORGET REALI:")
print("=" * 60)

for i in range(5):
    rand_index = random.randint(0, len(train_dataset) - 1)
    example = train_dataset[rand_index]
    question = example["question"]
    paraphrased_question = example["paraphrased_question"]
    expected_answer = example["answer"]

    inputs = tokenizer(paraphrased_question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_only = model_answer[len(paraphrased_question) :].strip()

    print(f"\n--- Test {i + 1} ---")
    print(f"❓ Domanda: {question}")
    print(f"❓ Domanda parafrasata: {paraphrased_question}")
    print(f"📚 Risposta attesa: {expected_answer[:80]}...")
    print(f"🤖 Risposta modello: {response_only}")
    print("-" * 60)

print("🧪 TEST CON DOMANDE RETAIN REALI:")
for i in range(5):
    rand_index = random.randint(0, len(retain_dataset) - 1)
    example = retain_dataset[rand_index]
    question = example["question"]
    expected_answer = example["answer"]

    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_only = model_answer[len(question) :].strip()

    print(f"\n--- Test {i + 1} ---")
    print(f"❓ Domanda: {question}")
    print(f"📚 Risposta attesa: {expected_answer[:80]}...")
    print(f"🤖 Risposta modello: {response_only}")
    print("-" * 60)
