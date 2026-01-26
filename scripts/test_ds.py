# test_dataset_sizes.py
from datasets import load_dataset

print("🔍 Verifica dimensioni dataset TOFU:")

# Carica i dataset
forget_data = load_dataset("locuslab/TOFU", "forget10", split="train")
retain_data = load_dataset("locuslab/TOFU", "retain90", split="train")

print(f"✅ Forget10: {len(forget_data)} esempi")
print(f"✅ Retain90: {len(retain_data)} esempi")
print(f"✅ Totale: {len(forget_data) + len(retain_data)} esempi")

# Verifica che non ci siano sovrapposizioni
print("\n📋 Primi 3 esempi forget:")
for i in range(3):
    print(f"  {i}: {forget_data[i]['question'][:50]}...")

print("\n📋 Primi 3 esempi retain:")
for i in range(3):
    print(f"  {i}: {retain_data[i]['question'][:50]}...")
