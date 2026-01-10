import os
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from datasets import load_dataset

# --- PARAMETERS ---
N_GRAM = 15
MAX_EXAMPLES_TO_PRINT = 10

# --- LOAD DATASETS ---
print("Loading datasets...")
ds_polaris = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train", streaming=True)
ds_math500 = load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True)
ds_aime2024 = load_dataset("HuggingFaceH4/aime_2024", split="train", streaming=True)
ds_aime2025 = load_dataset("MathArena/aime_2025", split="train", streaming=True)
ds_openthoughts = load_dataset("open-thoughts/OpenThoughts2-1M", split="train", streaming=True)

# --- UTILITIES ---
# Below is decontamination using n-grams of tokens
# def get_ngrams(text, n):
#     tokens = text.split()
#     return set([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else [])

# Below is decontamination using n-grams of words
def get_ngrams(text, n):
    tokens = [t.lower() for t in word_tokenize(text)]
    if len(tokens) < n:
        return set()
    return set([' '.join(ng) for ng in ngrams(tokens, n)])


def add_problem_ngrams(ds, name, global_dict, ngram_to_prompt):
    """Collect word-based n-grams and remember original text."""
    count = 0
    for ex in tqdm(ds, desc=f"Building n-grams for {name}", leave=False):
        problem = ex.get("problem")
        if not problem:
            continue
        count += 1
        for ng in get_ngrams(problem, N_GRAM):
            global_dict[name].add(ng)
            if ng not in ngram_to_prompt:
                # Use hash to identify RL prompt uniquely
                ngram_to_prompt[ng] = (name, hash(problem), problem)
    print(f"Collected {count:,} problems from {name}")
    return count

# --- BUILD RL N-GRAM SETS ---
all_rl_ngrams = {
    "Polaris": set(),
    "MATH-500": set(),
    "AIME-2024": set(),
    "AIME-2025": set(),
}
ngram_to_prompt = {}  # map ngram → (dataset_name, rl_prompt_id, full_text)

print("Building n-gram sets from RL datasets...")
count_polaris = add_problem_ngrams(ds_polaris, "Polaris", all_rl_ngrams, ngram_to_prompt)
count_math500 = add_problem_ngrams(ds_math500, "MATH-500", all_rl_ngrams, ngram_to_prompt)
count_aime2024 = add_problem_ngrams(ds_aime2024, "AIME-2024", all_rl_ngrams, ngram_to_prompt)
count_aime2025 = add_problem_ngrams(ds_aime2025, "AIME-2025", all_rl_ngrams, ngram_to_prompt)

total_rl_prompts = count_polaris + count_math500 + count_aime2024 + count_aime2025
print(f"Total RL problem prompts: {total_rl_prompts:,}")
print(f"Total unique RL n-grams collected: {len(ngram_to_prompt):,}")

# Combined n-gram union for fast overlap lookup
all_rl_ngram_union = set().union(*all_rl_ngrams.values())

# --- CONTAMINATION CHECK ---
def extract_user_prompt(conversation):
    if isinstance(conversation, list):
        for msg in conversation:
            if isinstance(msg, dict) and msg.get("from") == "user" and "value" in msg:
                return msg["value"]
    elif isinstance(conversation, dict):
        if conversation.get("from") == "user":
            return conversation.get("value")
    elif isinstance(conversation, str):
        return conversation
    return None

def find_overlap_and_source(user_prompt):
    if not user_prompt:
        return False, None, None, None, None
    sft_ngrams = get_ngrams(user_prompt, N_GRAM)
    overlap = sft_ngrams & all_rl_ngram_union
    if not overlap:
        return False, None, None, None, None
    example_ng = next(iter(overlap))
    ds_name, rl_prompt_id, rl_prompt_text = ngram_to_prompt.get(example_ng, ("Unknown", None, None))
    return True, " ".join(example_ng.split()), ds_name, rl_prompt_id, rl_prompt_text

# --- FILTERING LOOP ---
print("\nFiltering OpenThoughts SFT dataset (linear scan)...")
total_sft = 0
kept_sft = 0
contaminated_examples = []

# Track unique RL prompt IDs contaminated per dataset
contaminated_rl_prompts = {name: set() for name in all_rl_ngrams.keys()}

with open("filtered_openthoughts2.jsonl", "w", encoding="utf-8") as f_out:
    for ex in tqdm(ds_openthoughts, desc="Filtering SFT"):
        total_sft += 1
        conversation = ex.get("conversations")
        user_prompt = extract_user_prompt(conversation)

        contaminated, overlap_str, ds_name, rl_prompt_id, rl_prompt = find_overlap_and_source(user_prompt)
        if contaminated:
            if ds_name in contaminated_rl_prompts and rl_prompt_id is not None:
                contaminated_rl_prompts[ds_name].add(rl_prompt_id)
            if len(contaminated_examples) < MAX_EXAMPLES_TO_PRINT:
                contaminated_examples.append({
                    "sft_prompt": user_prompt,
                    "matched_dataset": ds_name,
                    "overlap_ngram": overlap_str,
                    "rl_prompt": rl_prompt,
                })
            continue
        kept_sft += 1
        f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")

# --- STATS ---
contaminated_count = total_sft - kept_sft
contamination_pct = (contaminated_count / total_sft * 100) if total_sft else 0

print("\n=== Filtering Statistics ===")
print(f"Total RL problems: {total_rl_prompts:,}")
print(f"Total SFT examples: {total_sft:,}")
print(f"Remaining after filtering: {kept_sft:,}")
print(f"Filtered (contaminated): {contaminated_count:,} ({contamination_pct:.2f}%)\n")

print("Contamination by unique RL problems:")
for name, ids in contaminated_rl_prompts.items():
    print(f"  {name:10s}: {len(ids):6,} unique RL prompts found in SFT")

# --- EXAMPLE PRINTING ---
if contaminated_examples:
    print("\n=== Example Contaminations (up to 10) ===")
    for i, ex in enumerate(contaminated_examples, 1):
        print(f"\nExample {i}:")
        print(f"Matched dataset: {ex['matched_dataset']}")
        print(f"Overlapping n-gram: {ex['overlap_ngram']}")
        print(f"\nSFT Prompt:\n{ex['sft_prompt'][:500]}")
        print(f"\nRL Prompt (from {ex['matched_dataset']}):\n{ex['rl_prompt'][:500]}")
else:
    print("\nNo contaminated examples found.")

print("\nDone. Filtered dataset saved to filtered_openthoughts2.jsonl")
