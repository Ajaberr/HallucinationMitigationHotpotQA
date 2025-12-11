
import json
import math

path = "tinker_sem_entropy_abstention_results.json"
with open(path, 'r') as f:
    data = json.load(f)

abstentions = [x for x in data if x['is_abstention_decision']]
answers = [x for x in data if not x['is_abstention_decision']]

print(f"Total: {len(data)}")
print(f"Abstentions: {len(abstentions)} ({len(abstentions)/len(data)*100:.1f}%)")

def get_stats(lst):
    if not lst: return 0.0, 0.0
    mean = sum(lst) / len(lst)
    sorted_lst = sorted(lst)
    median = sorted_lst[len(lst)//2]
    return mean, median

idk_entropies = [x['entropy'] for x in abstentions]
ans_entropies = [x['entropy'] for x in answers]

idk_mean, idk_med = get_stats(idk_entropies)
ans_mean, ans_med = get_stats(ans_entropies)

print(f"\nAbstention Entropy (Mean): {idk_mean:.4f}")
print(f"Abstention Entropy (Median): {idk_med:.4f}")
print(f"Answer Entropy (Mean): {ans_mean:.4f}")

print("\nSample Abstention Clusters:")
for x in abstentions[:3]:
    print(f"Q: {x['question'][:30]}...")
    print(f"Ent: {x['entropy']:.4f}")
    if 'clusters' in x:
        texts = [c['text'] for c in x['clusters']]
        print(f"Clusters: {texts}")
    print("-" * 20)
