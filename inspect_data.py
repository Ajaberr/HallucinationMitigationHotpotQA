from datasets import load_dataset

ds = load_dataset("fsiddiqui2/hotpotqa-abstention-70-30", split="train", streaming=True)

print("--- INSPECTING DATASET ---")
for i, item in enumerate(ds):
    if i >= 5: break
    print(f"\nItem {i}:")
    print(f"Target: {repr(item['target'])}")
    if 'short_target' in item:
        print(f"Short Target: {repr(item['short_target'])}")
