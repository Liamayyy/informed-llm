from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm

from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO, LABELS
from data_loader import ds
from model import Model, ModelPipeline, AncCtx

# ──────────────────────────────────────────────────────────────────────────
# Helper transform
# ──────────────────────────────────────────────────────────────────────────
def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]


# ──────────────────────────────────────────────────────────────────────────
# Role prompts
# ──────────────────────────────────────────────────────────────────────────
_DEBATER_ROLES = {
    "SUPPORTS": (
        "You are a fact‐checking debater. Argue that the claim is SUPPORTS. "
        "Explain why this claim is supported in 1–3 paragraphs."
    ),
    "REFUTES": (
        "You are a fact‐checking debater. Argue that the claim is REFUTES. "
        "Explain why this claim is refuted in 1–3 paragraphs."
    ),
    "NOT ENOUGH INFO": (
        "You are a fact‐checking debater. Argue that the claim is NOT ENOUGH INFO. "
        "Explain why we lack sufficient evidence in 1–3 paragraphs."
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Base classifier & simple debate models
# ──────────────────────────────────────────────────────────────────────────
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
    enforce_labels=True,
    instructions="You are an impartial fact‐checking judge. Read the arguments and choose one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Support arguments:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Refute arguments:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Insufficient arguments:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "Return exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    ),
    name="judge",
)


# ──────────────────────────────────────────────────────────────────────────
# Response‐1 models
# ──────────────────────────────────────────────────────────────────────────
RESP1_SUP = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-SUPPORTS",
    instructions="You are a SUPPORTS debater. Respond defending your stance in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "As the SUPPORTS debater, respond to these initial arguments."
    ),
)
RESP1_REF = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-REFUTES",
    instructions="You are a REFUTES debater. Respond defending your stance in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "As the REFUTES debater, respond to these initial arguments."
    ),
)
RESP1_NEI = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-NOT ENOUGH INFO",
    instructions="You are a NOT ENOUGH INFO debater. Respond defending your stance in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "As the NOT ENOUGH INFO debater, respond to these initial arguments."
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# Response‐2 models
# ──────────────────────────────────────────────────────────────────────────
RESP2_SUP = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-SUPPORTS",
    instructions="You are a SUPPORTS debater. Respond again to the first round of replies in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        "As the SUPPORTS debater, reply to these first responses."
    ),
)
RESP2_REF = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-REFUTES",
    instructions="You are a REFUTES debater. Respond again to the first round of replies in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        "As the REFUTES debater, reply to these first responses."
    ),
)
RESP2_NEI = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-NOT ENOUGH INFO",
    instructions="You are a NOT ENOUGH INFO debater. Respond again to the first round of replies in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        "As the NOT ENOUGH INFO debater, reply to these first responses."
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# Closing remarks
# ──────────────────────────────────────────────────────────────────────────
CLOS_SUP = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-SUPPORTS",
    instructions="You are a SUPPORTS debater. Provide closing remarks summarizing your key points.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp2 SUPPORTS:\n{ctx.get('Model(response2-SUPPORTS)','')}\n\n"
        "Now give your closing remarks."
    ),
)
CLOS_REF = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-REFUTES",
    instructions="You are a REFUTES debater. Provide closing remarks summarizing your key points.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp2 REFUTES:\n{ctx.get('Model(response2-REFUTES)','')}\n\n"
        "Now give your closing remarks."
    ),
)
CLOS_NEI = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-NOT ENOUGH INFO",
    instructions="You are a NOT ENOUGH INFO debater. Provide closing remarks summarizing your key points.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        f"Resp2 NOT ENOUGH INFO:\n{ctx.get('Model(response2-NOT ENOUGH INFO)','')}\n\n"
        "Now give your closing remarks."
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# Final extended judge
# ──────────────────────────────────────────────────────────────────────────
JUDGE_EXT = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="You are an impartial fact‐checking judge. Read the entire debate—including all rounds and closing remarks—and choose one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        f"Resp2 SUPPORTS:\n{ctx.get('Model(response2-SUPPORTS)','')}\n\n"
        f"Resp2 REFUTES:\n{ctx.get('Model(response2-REFUTES)','')}\n\n"
        f"Resp2 NOT ENOUGH INFO:\n{ctx.get('Model(response2-NOT ENOUGH INFO)','')}\n\n"
        f"Closing SUPPORTS:\n{ctx.get('Model(closing-SUPPORTS)','')}\n\n"
        f"Closing REFUTES:\n{ctx.get('Model(closing-REFUTES)','')}\n\n"
        f"Closing NOT ENOUGH INFO:\n{ctx.get('Model(closing-NOT ENOUGH INFO)','')}\n\n"
        "Return exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    ),
    name="judge-extended",
)


# ──────────────────────────────────────────────────────────────────────────
# Additional Arguments model
# ──────────────────────────────────────────────────────────────────────────
ADDL_ARGS = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="additional-arguments",
    instructions=(
        "You are a fact‐checking debater. Having seen two rounds of arguments and responses, "
        "now provide a fresh set of additional arguments in 1–2 paragraphs."
    ),
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Resp2 SUPPORTS:\n{ctx.get('Model(response2-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Resp2 REFUTES:\n{ctx.get('Model(response2-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        f"Resp2 NOT ENOUGH INFO:\n{ctx.get('Model(response2-NOT ENOUGH INFO)','')}\n\n"
        "Now provide additional arguments."
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline wiring
# ──────────────────────────────────────────────────────────────────────────

# Base pipeline
P0 = Model(repo=None, input_transform=_normalize, name="normaliser")
P0 >> BASE
PIPE_BASE = ModelPipeline([P0])

# Simple 3-way debate
P3 = Model(repo=None, input_transform=_normalize, name="normaliser")
P3 >> (DEB_SUP, DEB_REF, DEB_NEI)
P3 >> JUDGE
PIPE_DEBATE = ModelPipeline([P3])

# Extended debate pipeline
P4 = Model(repo=None, input_transform=_normalize, name="normaliser")
# initial arguments
P4 >> (DEB_SUP, DEB_REF, DEB_NEI)
# first responses
P4 >> (RESP1_SUP, RESP1_REF, RESP1_NEI)
# second responses
P4 >> (RESP2_SUP, RESP2_REF, RESP2_NEI)
# closing remarks
P4 >> (CLOS_SUP, CLOS_REF, CLOS_NEI)
# final judge
P4 >> JUDGE_EXT
PIPE_DEBATE_EXT = ModelPipeline([P4])


# Extended debate *with* additional-arguments and repeated rounds
P5 = Model(repo=None, input_transform=_normalize, name="normaliser")

# 1) Initial arguments
P5 >> (DEB_SUP, DEB_REF, DEB_NEI)
# 2) First responses
P5 >> (RESP1_SUP, RESP1_REF, RESP1_NEI)
# 3) Second responses
P5 >> (RESP2_SUP, RESP2_REF, RESP2_NEI)
# 4) Inject fresh material
P5 >> ADDL_ARGS
# 5) Re-run initial arguments
P5 >> (DEB_SUP, DEB_REF, DEB_NEI)
# 6) Re-run first responses
P5 >> (RESP1_SUP, RESP1_REF, RESP1_NEI)
# 7) Re-run second responses
P5 >> (RESP2_SUP, RESP2_REF, RESP2_NEI)
# 8) Closing remarks
P5 >> (CLOS_SUP, CLOS_REF, CLOS_NEI)
# 9) Final judge
P5 >> JUDGE_EXT

PIPE_DEBATE_EXT2 = ModelPipeline([P5])

PIPELINES = {
    "base": PIPE_BASE,
    "debate-3": PIPE_DEBATE,
    "debate-extended": PIPE_DEBATE_EXT,
    "debate-extended2": PIPE_DEBATE_EXT2,
}


# ──────────────────────────────────────────────────────────────────────────
# Benchmarking loop (unchanged)
# ──────────────────────────────────────────────────────────────────────────
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
