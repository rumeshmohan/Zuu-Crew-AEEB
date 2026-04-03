import json
import re
import os

# Robust path handling
base_dir = os.path.dirname(os.path.abspath(__file__))
jsonl_path = os.path.join(base_dir, '..', 'artifacts', 'data', 'intern_predictions.jsonl')

print("🔍 Finding differences between Intern and Truth...\n")

hallucinations = []

if not os.path.exists(jsonl_path):
    print(f"❌ Error: File not found at {jsonl_path}")
    exit()

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        
        q = entry.get('question', '')
        gt = entry.get('ground_truth', '')
        pred = entry.get('intern_answer', '')
        
        # Logic: Flag if the prediction doesn't exactly match the truth
        # and contains numbers (common area for hallucinations)
        if gt.strip() != pred.strip():
            if re.search(r'\d', q + gt + pred):
                hallucinations.append({
                    'q': q,
                    'truth': gt,
                    'pred': pred
                })

# Print top 3 examples of mismatches
for i, h in enumerate(hallucinations[:3], 1):
    print(f"{'='*70}")
    print(f"MISMATCH EXAMPLE {i}")
    print(f"{'='*70}")
    print(f"\nQ: {h['q']}")
    print(f"\n✅ Truth: {h['truth']}")
    print(f"\n❌ Intern: {h['pred']}\n")

print(f"✅ Found {len(hallucinations)} cases where the Intern's answer differed from the Truth.")