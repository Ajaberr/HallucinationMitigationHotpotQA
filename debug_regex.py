import re

test_cases = [
    "1838. --> 1838",
    "Nancy Sinatra. --> Nancy Sinatra",
    "Wolf Alice. --> Wolf Alice",
    "Rome. --> Rome",
    "yes. --> yes",
    "Step 1: The question asks... \nStep 2: ... --> Answer" # Hypothetical
]

print(f"Testing regex: r'\\s*-{{2,}}>\\s*'")

for curr in test_cases:
    print(f"\nScanning: '{curr}'")
    
    # 1. Search
    arrow_match = re.search(r"\s*-{2,}>\s*", curr)
    if arrow_match:
        print(f"  [MATCH FOUND]")
        # 2. Split
        parts = re.split(r"\s*-{2,}>\s*", curr)
        print(f"  Parts: {parts}")
        parsed = parts[-1].strip()
        print(f"  Parsed: '{parsed}'")
        
        # Check against expectation
        expected = curr.split("-->")[-1].strip()
        if parsed == expected:
             print("  [SUCCESS] Matches expectation.")
        else:
             print("  [DIFF] expectation mismatch.")

    else:
        print("  [NO MATCH] Regex failed.")
