
import re
import string

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

REFUSAL_PHRASE = "i dont know"

def check_abstention_robust(prediction):
    norm_pred = normalize_answer(prediction)
    return REFUSAL_PHRASE in norm_pred

def check_abstention_weak(text):
    return "i dont know" in text.lower().replace("'", "")

test_cases = [
    "I don't know.",
    "i don't know",
    "I really don't know",
    "Sorry, I don't know the answer.",
    "I do not know", # unexpected?
    "I dont know",
    "I don't know!",
    "I don't  know" 
]

print(f"{'Text':<35} | {'Robust':<10} | {'Weak':<10} | {'Match?'}")
print("-" * 70)
for t in test_cases:
    r = check_abstention_robust(t)
    w = check_abstention_weak(t)
    print(f"{t:<35} | {str(r):<10} | {str(w):<10} | {r==w}")
