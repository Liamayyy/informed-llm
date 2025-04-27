from __future__ import annotations

import logging

from tqdm.auto import tqdm

from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO, LABELS
from model import AncCtx, Model, ModelPipeline
from data_loader import ds
from tqdm.auto import tqdm

# ─────────────────────────── COMPLEX EXAMPLE GRAPH ─────────────────────

def normalise(_: AncCtx, txt: str) -> str:
    clean = txt.strip()
    return clean[:1].upper() + clean[1:]

T0 = Model(repo=None, input_transform=normalise, name="normaliser")

M1 = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    instructions="Label the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    name="clf-no-search",
)
M2 = Model(
    MODEL_REPO_DEFAULT,
    use_search=True,
    instructions="Label the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    name="clf-with-search",
)

def majority(ctx: AncCtx, _orig: str) -> str:
    votes = [v for k, v in ctx.items() if k.startswith("Model(") and v in LABELS]
    if not votes:
        return "NOT ENOUGH INFO"
    return max(set(votes), key=votes.count)

A3 = Model(repo=None, input_transform=majority, name="majority-vote", enforce_labels=True)

def parents_to_prompt(ctx: AncCtx, orig: str) -> str:
    lab1 = ctx.get(repr(M1), "?")
    lab2 = ctx.get(repr(M2), "?")
    maj = ctx.get(repr(A3), "?")
    return (
        f"Claim: {orig}\n"
        f"Classifier-1 label: {lab1}\nClassifier-2 label: {lab2}\n"
        f"Majority decision: {maj}\n"
        "Provide a concise explanation (1–2 sentences) justifying the majority label."
    )

E4 = Model(
    EXPLAINER_REPO,
    use_search=False,
    instructions="You are an expert fact-checker.",
    enforce_labels=False,
    input_transform=parents_to_prompt,
    name="explainer",
)

F5 = Model(repo=None, input_transform=lambda ctx, _o: ctx.get(repr(E4), ""), name="verdict")

# DSL wiring
T0 >> (M1, M2)
M1 >> A3
M2 >> A3
A3 >> E4
E4 >> F5

pipelines = [ModelPipeline([T0])]

# ─────────────────────────── LOGGING & BENCHMARK ────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        filename="benchmark.log",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)
    results = []
    for pipe in pipelines:
        name = repr(pipe)
        print(f"\nEvaluating pipeline: {name}")
        correct = 0
        for ex in tqdm(ds, desc=name):
            claim = ex["claim"].strip()
            ref = ex["label"].strip().upper()
            pred = pipe.predict_label(claim)
            correct += int(pred == ref)
            logger.info(
                f"Pipeline: {name}\nTree: {pipe.predict(claim)}\nRef: {ref}\n{'='*70}"
            )
        acc = correct / len(ds)
        results.append((name, acc))
        print(f"Root accuracy ({repr(M1)}): {acc:.3%}")
    print("\n=== Summary ===")
    for n, a in results:
        print(f"{n}: {a:.3%}")
