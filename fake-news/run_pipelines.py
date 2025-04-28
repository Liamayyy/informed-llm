from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import numpy as np

from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO
from data_loader import ds
from model import Model, ModelPipeline, AncCtx


# ────────────────────────── helpers ──────────────────────────

def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]


# ────────────────────────── debate-role prompts ──────────────────────────

_DEBATER_ROLES = {
    "FAKE": (
        "You are a fact-checking debater. Argue that the claim is FAKE. "
        "Explain in 1–3 paragraphs why the headline is fabricated or misleading."
    ),
    "TRUE": (
        "You are a fact-checking debater. Argue that the claim is TRUE. "
        "Explain in 1–3 paragraphs with evidence why the headline is accurate."
    ),
}


# ────────────────────────── model definitions ──────────────────────────

BASE = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    enforce_labels=True,
    instructions="Label the claim as FAKE or TRUE.",
    name="base-clf",
)

DEB_FAKE = Model(
    MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["FAKE"], enforce_labels=False, name="debater-FAKE"
)
DEB_TRUE = Model(
    MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["TRUE"], enforce_labels=False, name="debater-TRUE"
)

JUDGE = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="You are an impartial fact-checking judge. Choose FAKE or TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"FAKE arguments:\n{ctx.get('Model(debater-FAKE)','')}\n\n"
        f"TRUE arguments:\n{ctx.get('Model(debater-TRUE)','')}\n\n"
        "Label:"
    ),
    name="judge",
)

# First and second responses
RESP1_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-FAKE",
    instructions="Respond defending FAKE in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"FAKE:\n{ctx.get('Model(debater-FAKE)','')}\n\n"
        f"TRUE:\n{ctx.get('Model(debater-TRUE)','')}\n"
    ),
)

RESP1_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-TRUE",
    instructions="Respond defending TRUE in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"FAKE:\n{ctx.get('Model(debater-FAKE)','')}\n\n"
        f"TRUE:\n{ctx.get('Model(debater-TRUE)','')}\n"
    ),
)

RESP2_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-FAKE",
    instructions="Second response for FAKE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Resp1 FAKE:\n{ctx.get('Model(response1-FAKE)','')}\n\n"
        f"Resp1 TRUE:\n{ctx.get('Model(response1-TRUE)','')}\n"
    ),
)

RESP2_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-TRUE",
    instructions="Second response for TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Resp1 FAKE:\n{ctx.get('Model(response1-FAKE)','')}\n\n"
        f"Resp1 TRUE:\n{ctx.get('Model(response1-TRUE)','')}\n"
    ),
)

# Closing remarks
CLOS_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-FAKE",
    instructions="Closing remarks for FAKE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n{ctx.get('Model(response2-FAKE)','')}\n"
    ),
)

CLOS_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-TRUE",
    instructions="Closing remarks for TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n{ctx.get('Model(response2-TRUE)','')}\n"
    ),
)

# Fresh arguments model
ADDL_ARGS = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="additional-arguments",
    instructions="Provide new arguments after two full rounds.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Earlier FAKE:\n{ctx.get('Model(response2-FAKE)','')}\n\n"
        f"Earlier TRUE:\n{ctx.get('Model(response2-TRUE)','')}\n"
    ),
)

# Final judge after full debate
JUDGE_EXT = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="Final judge decision after full debate. Choose FAKE or TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"{ctx.get('Model(closing-FAKE)','')}\n"
        f"{ctx.get('Model(closing-TRUE)','')}\n"
        "Label:"
    ),
    name="judge-extended",
)


# ────────────────────────── pipeline wiring ──────────────────────────

P0 = Model(repo=None, input_transform=_normalize, name="normaliser")
P0 >> BASE
PIPE_BASE = ModelPipeline([P0])

P1 = Model(repo=None, input_transform=_normalize, name="normaliser")
P1 >> (DEB_FAKE, DEB_TRUE)
P1 >> JUDGE
PIPE_DEBATE = ModelPipeline([P1])

P2 = Model(repo=None, input_transform=_normalize, name="normaliser")
P2 >> (DEB_FAKE, DEB_TRUE)
P2 >> (RESP1_FAKE, RESP1_TRUE)
P2 >> (RESP2_FAKE, RESP2_TRUE)
P2 >> (CLOS_FAKE, CLOS_TRUE)
P2 >> JUDGE_EXT
PIPE_DEBATE_EXT = ModelPipeline([P2])

P3 = Model(repo=None, input_transform=_normalize, name="normaliser")
P3 >> (DEB_FAKE, DEB_TRUE)
P3 >> (RESP1_FAKE, RESP1_TRUE)
P3 >> (RESP2_FAKE, RESP2_TRUE)
P3 >> ADDL_ARGS
P3 >> (DEB_FAKE, DEB_TRUE)
P3 >> (RESP1_FAKE, RESP1_TRUE)
P3 >> (RESP2_FAKE, RESP2_TRUE)
P3 >> (CLOS_FAKE, CLOS_TRUE)
P3 >> JUDGE_EXT
PIPE_DEBATE_EXT2 = ModelPipeline([P3])

PIPELINES = {
    "base":             PIPE_BASE,
    "debate-2":         PIPE_DEBATE,
    "debate-extended":  PIPE_DEBATE_EXT,
    "debate-extended2": PIPE_DEBATE_EXT2,
}


# ────────────────────────── benchmarking ──────────────────────────

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
        y_true = []
        y_pred = []

        print(f"Evaluating pipeline: {name}")

        for ex in tqdm(ds, desc=name):
            claim = ex["claim"].strip()
            ref = _norm(ex["label"])

            tree, raw_pred = pipe.predict_with_label(claim)
            pred = _norm(raw_pred)

            if pred == ref:
                correct += 1

            y_true.append(ref)
            y_pred.append(pred)

            logger.info(
                f"Pipeline: {name}\n"
                f"Predicted: {pred}\n"
                f"Tree: {tree}\n"
                f"Ref: {ref}\n"
                + "=" * 70
            )

        accuracy = correct / len(ds)
        print(f" → accuracy: {accuracy:.3%}")

        # Additional metrics
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=["FAKE", "TRUE"]))
        print("\n" + "-"*80 + "\n")

    print("Done.")
