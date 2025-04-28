from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm

from config      import MODEL_REPO_DEFAULT, EXPLAINER_REPO
from data_loader import ds
from model       import Model, ModelPipeline, AncCtx


# ────────────────────────── helpers ──────────────────────────
def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]


# ─────────────── debate-role system prompts (2-way) ───────────────
_DEBATER_ROLES = {
    "FAKE": (
        "You are a fact-checking debater. Argue that the claim is FAKE. "
        "Explain, in 1–3 paragraphs, why the headline is fabricated or misleading."
    ),
    "TRUE": (
        "You are a fact-checking debater. Argue that the claim is TRUE. "
        "Explain, in 1–3 paragraphs, with evidence, why the headline is accurate."
    ),
}


# ─────────────── base classifier ───────────────
BASE = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    enforce_labels=True,
    instructions="Label the claim as FAKE or TRUE.",
    name="base-clf",
)


# ─────────────── debaters ───────────────
DEB_FAKE = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["FAKE"],
    enforce_labels=False,
    name="debater-FAKE",
)
DEB_TRUE = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["TRUE"],
    enforce_labels=False,
    name="debater-TRUE",
)

# ─────────────── judge ───────────────
JUDGE = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions=(
        "You are an impartial fact-checking judge. Read the arguments and "
        "output exactly one label: FAKE or TRUE."
    ),
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"FAKE arguments:\n{ctx.get('Model(debater-FAKE)', '')}\n\n"
        f"TRUE arguments:\n{ctx.get('Model(debater-TRUE)', '')}\n\n"
        "Label:"
    ),
    name="judge",
)

# ─────────────── first-response round ───────────────
RESP1_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-FAKE",
    instructions="You are the FAKE debater. Respond to the opposing side in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Initial FAKE:\n{ctx.get('Model(debater-FAKE)', '')}\n\n"
        f"Initial TRUE:\n{ctx.get('Model(debater-TRUE)', '')}\n\n"
        "Your reply:"
    ),
)

RESP1_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-TRUE",
    instructions="You are the TRUE debater. Respond to the opposing side in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Initial FAKE:\n{ctx.get('Model(debater-FAKE)', '')}\n\n"
        f"Initial TRUE:\n{ctx.get('Model(debater-TRUE)', '')}\n\n"
        "Your reply:"
    ),
)

# ─────────────── second-response round ───────────────
RESP2_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-FAKE",
    instructions="You are the FAKE debater. Respond again in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Resp1 FAKE:\n{ctx.get('Model(response1-FAKE)', '')}\n\n"
        f"Resp1 TRUE:\n{ctx.get('Model(response1-TRUE)', '')}\n\n"
        "Your reply:"
    ),
)

RESP2_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-TRUE",
    instructions="You are the TRUE debater. Respond again in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Resp1 FAKE:\n{ctx.get('Model(response1-FAKE)', '')}\n\n"
        f"Resp1 TRUE:\n{ctx.get('Model(response1-TRUE)', '')}\n\n"
        "Your reply:"
    ),
)

# ─────────────── closing remarks ───────────────
CLOS_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-FAKE",
    instructions="You are the FAKE debater. Provide a concise closing statement.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Initial FAKE:\n{ctx.get('Model(debater-FAKE)', '')}\n\n"
        f"Resp1 FAKE:\n{ctx.get('Model(response1-FAKE)', '')}\n\n"
        f"Resp2 FAKE:\n{ctx.get('Model(response2-FAKE)', '')}\n\n"
        "Closing statement:"
    ),
)

CLOS_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-TRUE",
    instructions="You are the TRUE debater. Provide a concise closing statement.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Initial TRUE:\n{ctx.get('Model(debater-TRUE)', '')}\n\n"
        f"Resp1 TRUE:\n{ctx.get('Model(response1-TRUE)', '')}\n\n"
        f"Resp2 TRUE:\n{ctx.get('Model(response2-TRUE)', '')}\n\n"
        "Closing statement:"
    ),
)

# ─────────────── extended-round judge ───────────────
JUDGE_EXT = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions=(
        "You are an impartial fact-checking judge. Having read the entire "
        "debate—including all rounds and closing remarks—output exactly one "
        "label: FAKE or TRUE."
    ),
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"Initial FAKE:\n{ctx.get('Model(debater-FAKE)', '')}\n\n"
        f"Initial TRUE:\n{ctx.get('Model(debater-TRUE)', '')}\n\n"
        f"Resp1 FAKE:\n{ctx.get('Model(response1-FAKE)', '')}\n\n"
        f"Resp1 TRUE:\n{ctx.get('Model(response1-TRUE)', '')}\n\n"
        f"Resp2 FAKE:\n{ctx.get('Model(response2-FAKE)', '')}\n\n"
        f"Resp2 TRUE:\n{ctx.get('Model(response2-TRUE)', '')}\n\n"
        f"Closing FAKE:\n{ctx.get('Model(closing-FAKE)', '')}\n\n"
        f"Closing TRUE:\n{ctx.get('Model(closing-TRUE)', '')}\n\n"
        "Label:"
    ),
    name="judge-extended",
)


# ─────────────── pipeline wiring ───────────────
# Base pipeline
P0 = Model(repo=None, input_transform=_normalize, name="normaliser")
P0 >> BASE
PIPE_BASE = ModelPipeline([P0])

# Simple 2-way debate
P1 = Model(repo=None, input_transform=_normalize, name="normaliser")
P1 >> (DEB_FAKE, DEB_TRUE)
P1 >> JUDGE
PIPE_DEBATE = ModelPipeline([P1])

# Extended debate (2 rounds + closings)
P2 = Model(repo=None, input_transform=_normalize, name="normaliser")
P2 >> (DEB_FAKE, DEB_TRUE)
P2 >> (RESP1_FAKE, RESP1_TRUE)
P2 >> (RESP2_FAKE, RESP2_TRUE)
P2 >> (CLOS_FAKE, CLOS_TRUE)
P2 >> JUDGE_EXT
PIPE_DEBATE_EXT = ModelPipeline([P2])

PIPELINES = {
    "base":             PIPE_BASE,
    "debate-2":         PIPE_DEBATE,
    "debate-extended":  PIPE_DEBATE_EXT,
}


# ─────────────── benchmark loop ───────────────
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
            ref   = _norm(ex["label"])

            tree, raw_pred = pipe.predict_with_label(claim)
            pred = _norm(raw_pred)

            if pred == ref:
                correct += 1

            logger.info(
                f"Pipeline: {name}\n"
                f"Predicted: {pred}\n"
                f"Tree: {tree}\n"
                f"Ref: {ref}\n"
                + "=" * 70
            )

        accuracy = correct / len(ds)
        print(f" → accuracy: {accuracy:.3%}")

    print("Done.")
