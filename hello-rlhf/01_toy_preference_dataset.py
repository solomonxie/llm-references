# $ venv/bin/python 01_toy_preference_dataset.py
#
# Goal: RLHF starts from preference data -- pairs of (chosen, rejected)
# completions for the same prompt, where "chosen" is preferred by a human
# (or, here, a stand-in ground-truth rule so the whole series is checkable
# without a human in the loop). This builds that dataset: random short
# sequences over a 4-symbol vocabulary, "preferred" defined as containing
# more of the symbol 'A' -- an arbitrary but perfectly checkable rule.
# Step 1: A toy preference dataset -- (chosen, rejected) sequence pairs

import random

random.seed(0)

VOCAB = ["A", "B", "C", "D"]
SEQ_LEN = 5


def random_sequence() -> list[int]:
    return [random.randrange(len(VOCAB)) for _ in range(SEQ_LEN)]


def true_score(seq: list[int]) -> int:
    # The ground-truth "human preference" rule for this whole series: more
    # A's (index 0) is better. Nothing in the model ever sees this function
    # directly -- only comparisons derived from it.
    return sum(1 for tok in seq if tok == 0)


def seq_to_str(seq: list[int]) -> str:
    return "".join(VOCAB[t] for t in seq)


def build_preference_pairs(n_pairs: int) -> list[dict]:
    pairs = []
    while len(pairs) < n_pairs:
        a, b = random_sequence(), random_sequence()
        score_a, score_b = true_score(a), true_score(b)
        if score_a == score_b:
            continue  # no preference signal if tied
        chosen, rejected = (a, b) if score_a > score_b else (b, a)
        pairs.append({"chosen": chosen, "rejected": rejected})
    return pairs


dataset = build_preference_pairs(200)

print(f"built {len(dataset)} preference pairs\n")
for pair in dataset[:5]:
    c, r = pair["chosen"], pair["rejected"]
    print(f"  chosen={seq_to_str(c)} (score={true_score(c)})  "
          f"rejected={seq_to_str(r)} (score={true_score(r)})")

avg_margin = sum(true_score(p["chosen"]) - true_score(p["rejected"]) for p in dataset) / len(dataset)
print(f"\naverage score margin (chosen - rejected): {avg_margin:.2f}")
print("this dataset -- pairs plus which one won -- is all steps 2+ ever see;")
print("the true_score() rule itself stays hidden from every model trained on it.")
