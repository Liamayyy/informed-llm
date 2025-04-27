from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm

from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO, LABELS
from data_loader import ds
from model import Model, ModelPipeline, AncCtx

# Helper transform
def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]

# Debate prompts
_DEBATER_ROLES = {
    "SUPPORTS": (
        "You are a fact-checking debater. Argue that the claim is SUPPORTS. "
        "Explain why this claim is supported in 1-3 paragraphs."
    ),
    "REFUTES": (
        "You are a fact-checking debater. Argue that the claim is REFUTES. "
        "Explain why this claim is refuted in 1-3 paragraphs."
    ),
    "NOT ENOUGH INFO": (
        "You are a fact-checking debater. Argue that the claim is NOT ENOUGH INFO. "
        "Explain why we lack sufficient evidence in 1-3 paragraphs."
    ),
}

# Model instances
BASE = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    enforce_labels=True,
    instructions="Label the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    name="base-clf",
)
DEB_SUP = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["SUPPORTS"],
    enforce_labels=False,
    name="debater-SUPPORTS",
)
DEB_REF = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["REFUTES"],
    enforce_labels=False,
    name="debater-REFUTES",
)
DEB_NEI = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["NOT ENOUGH INFO"],
    enforce_labels=False,
    name="debater-NOT ENOUGH INFO",
)
JUDGE = Model(
    EXPLAINER_REPO,
    use_search=False,
    instructions="You are an impartial fact-checking judge. Read the arguments carefully and choose one of: SUPPORTS, REFUTES, NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Support arguments:\n{ctx.get('Model(debater-SUPPORTS)', '')}\n\n"
        f"Refute arguments:\n{ctx.get('Model(debater-REFUTES)', '')}\n\n"
        f"Insufficient arguments:\n{ctx.get('Model(debater-NOT ENOUGH INFO)', '')}\n\n"
        "Return exactly one label: SUPPORTS, REFUTES, NOT ENOUGH INFO."
    ),
    name="judge",
    enforce_labels=True,
)

# Pipeline wiring
P0 = Model(repo=None, input_transform=_normalize, name="normaliser")
P0 >> BASE
PIPE_BASE = ModelPipeline([P0])

P3 = Model(repo=None, input_transform=_normalize, name="normaliser")
P3 >> (DEB_SUP, DEB_REF, DEB_NEI)
P3 >> JUDGE
PIPE_DEBATE = ModelPipeline([P3])

PIPELINES = {
    "base": PIPE_BASE,
    "debate-3": PIPE_DEBATE,
}

# Benchmark
if __name__ == "__main__":
    logging.basicConfig(
        filename="benchmark.log",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    print("Running evaluation on", len(ds), "examples…")

    def _norm(label: Union[str, List[str]]) -> str:
        if isinstance(label, list):
            label = label[-1]
        return label.strip().rstrip(".").upper()

    for name, pipe in PIPELINES.items():
        correct = 0
        print(f"Evaluating pipeline: {name}")

        for ex in tqdm(ds, desc=name):
            claim = ex["claim"].strip()
            ref = _norm(ex["label"])

            # single call
            tree, raw_pred = pipe.predict_with_label(claim)
            pred = _norm(raw_pred)

            if pred == ref:
                correct += 1

            logger.info(
                f"Pipeline: {name}\n"
                f"Predicted: {pred}\n"
                f"Tree: {tree}\n"
                f"Ref: {ref}\n"
                + "="*70
            )

        accuracy = correct / len(ds)
        print(f" → accuracy: {accuracy:.3%}")

    print("Done.")