import json
import numpy as np

file_path = "tinker/results/tinker_sem_entropy_abstention_results.json"

with open(file_path, 'r') as f:
    data = json.load(f)

print(f"Total items: {len(data)}")

abstentions = [x for x in data if x['is_abstention_decision']]
answered = [x for x in data if not x['is_abstention_decision']]

print(f"Abstentions: {len(abstentions)} ({len(abstentions)/len(data)*100:.2f}%)")
print(f"Answered: {len(answered)} ({len(answered)/len(data)*100:.2f}%)")

# Calculate metrics manually from the 'metrics' field
all_ems = [x['metrics']['em'] for x in data]
all_f1s = [x['metrics']['f1'] for x in data]

ans_ems = [x['metrics']['em'] for x in answered]
ans_f1s = [x['metrics']['f1'] for x in answered]

print("-" * 20)
print(f"Overall EM: {np.mean(all_ems):.4f}")
print(f"Overall F1: {np.mean(all_f1s):.4f}")
print("-" * 20)
print(f"Selective EM (Answered only): {np.mean(ans_ems):.4f}")
print(f"Selective F1 (Answered only): {np.mean(ans_f1s):.4f}")
