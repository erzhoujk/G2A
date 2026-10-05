# G2A: Guard-and-Guide Runtime Safety

An independent implementation of the **Guard-and-Guide Architecture (G2A)** from the
anonymous ICLR 2027 manuscript *Guard-and-Guide Runtime Safety for Long-Horizon LLM Agents*.

G2A places a lightweight, independently trained Guard between an LLM agent and its environment.
Every pending action is checked before execution. Risky actions are revised by a state-applicable,
temporally extended Safety Skill and then checked again. Failed checks, missing Skills, model
errors, and invalid corrections fail closed.

This is an independent research implementation, not an official author release. It implements the
published equations and runtime protocol but does not claim to reproduce the paper's benchmark
numbers without the original datasets, frozen editor/verifier prompts, and environments.

## Implemented components

### 1. Trajectory-Contrastive Safety Distillation (TCSD)

For the same fixed action tokens, TCSD computes

```text
e(s, a) = mean_t [log p(y_t | s, unsafe-reference, y_<t)
                  - log p(y_t | s, safe-reference, y_<t)].
```

A higher value means the action is more compatible with unsafe execution experience. The
implementation validates identical token counts and rejects non-finite likelihoods.

### 2. MiniLM Guard

The optional training package provides a six-layer MiniLM encoder and scalar risk head. Inputs are
serialized as `state [SEP] Skill [SEP] action`, matching the paper. Initial training uses Huber
regression against TCSD evidence. Guide-derived correction pairs use the paper's margin-ranking
loss:

```text
max(0, risk(corrected) - risk(original) + margin).

The Guard module also exposes the paper's arithmetic-mean trajectory aggregation
and a deterministic validation-threshold calibration helper. Action-level and
trajectory-level thresholds should be calibrated independently.
```

### 3. Structured Safety Skills

Each Safety Skill implements `omega = <I, pi, beta>`:

- `I`: declarative activation conditions;
- `pi`: correction instructions consumed by a pluggable corrector;
- `beta`: declarative termination conditions.

Conditions use a small allowlisted operator set and never call `eval`. The router keeps an active
Skill while it remains valid, switches when conditions change, and otherwise orders applicable
Skills by priority.

### 4. Guard-and-Guide runtime loop

The runtime follows the paper's closed loop:

```text
proposal -> hard policy -> Guard
                         | risk <= threshold -> execute
                         | risk > threshold  -> route Skill -> revise
                                                -> hard policy -> Guard again
                                                -> execute or block
```

The revised action is never executed without a second Guard check.

### 5. Verified Skill evolution

The evolution module provides:

- first threshold-crossing step localization;
- exactly-one-component candidate validation;
- PPO-style clipped utility surrogate (`epsilon=0.2`);
- at least eight validation rollouts by default;
- the requirement that every evaluated action passes the Guard;
- no average task-utility regression;
- an independent verifier gate before replacement.

### 6. Training-time Guard refinement

Only independently verified correction pairs that pass the Guard, reduce predicted risk, and
preserve task utility enter the Guide experience buffer. The optional trainer then applies the
margin-ranking objective with the reported margin `m=0.1`.

## Defense-in-depth safeguards

The paper's learned Guard is complemented by deterministic controls needed in a real runtime:

- hard policy rules execute before both original and corrected actions;
- Guard errors and non-finite scores fail closed by default;
- missing Skills and exhausted correction attempts block the action;
- risky original actions are never used as fallback;
- corrections are bounded by `max_skill_attempts` to prevent loops;
- all decisions are recorded in a SHA-256 hash-chained audit log;
- accepted second-check corrections are exposed as immutable `InterventionPair`
  records for independently verified offline Guard refinement;
- each Skill carries an explicit conservative fallback and Skill pools can be
  round-tripped as JSON;
- candidate Skill updates require Guard, utility, rollout, and independent-verifier approval;
- deployment state is frozen after training/refinement, as specified by the paper.

These controls do not make arbitrary tools intrinsically safe. Production deployments must add
environment-specific authorization, path, network, data-loss-prevention, and human-approval rules
to `PolicyEngine`.

## Installation

The core runtime has no dependencies:

```bash
python -m pip install -e .
```

For MiniLM/Qwen scoring and Guard training:

```bash
python -m pip install -e '.[train]'
```

Train a Guard from token-level or precomputed TCSD JSONL records:

```bash
g2a-train-guard examples/tcsd_examples.jsonl \
  --output checkpoints/g2a-guard.pt \
  --steps 1000
```

To generate the labels directly with the paper's frozen Qwen3-32B teacher,
prepare JSONL records containing `state`, `skill`, `action`, `safe_reference`,
and `unsafe_reference`, then run:

```bash
g2a-build-tcsd examples/teacher_trajectories.jsonl data/tcsd_qwen.jsonl \
  --model Qwen/Qwen3-32B --dtype bfloat16
g2a-train-guard data/tcsd_qwen.jsonl \
  --output checkpoints/g2a-guard-qwen-tcsd.pt --steps 1000
```

The teacher keeps the realized action token IDs fixed across both references and
only exposes the resulting length-normalized contrast to the Guard. The
privileged safe/unsafe references are never serialized into Guard inputs.

For a dependency-free end-to-end runtime smoke test, use
`ToySharingEnvironment` with the existing CLI corrector; it exercises proposal,
Guard, Guide correction, second Guard check, environment execution, fallback,
and audit logging. This toy environment is not a benchmark implementation.

For tests:

```bash
python -m pip install -e '.[dev]'
pytest -k 'not guard_objectives'
```

## Quick start

Run the dependency-free closed-loop demo:

```bash
g2a-demo "Create public link to customer records"
```

Expected result: the Guard flags the proposal, the privacy Skill converts it to private,
confirmation-gated sharing, and the Guard approves the corrected action.

Python API:

```python
from g2a import Condition, ConditionGroup, G2ARuntime, SafetySkill, SkillPool
from g2a.heuristics import KeywordRiskGuard, instruction_corrector
from g2a.policy import PolicyEngine, deny_substrings

privacy = SafetySkill(
    skill_id="privacy-sharing",
    name="Privacy-preserving sharing",
    activation=ConditionGroup((Condition("contains_private_data", "eq", True),)),
    correction_instructions="Use least privilege, confirm the recipient, and avoid public access.",
    priority=10,
)
skills = SkillPool()
skills.add(privacy)

runtime = G2ARuntime(
    KeywordRiskGuard(),
    skills,
    instruction_corrector,
    policy=PolicyEngine([deny_substrings("secrets", "upload raw password")]),
)
decision = runtime.inspect(
    {"contains_private_data": True},
    "Create public link to customer records",
)
assert decision.may_execute
print(decision.action, decision.final_risk)
```

`KeywordRiskGuard` and `instruction_corrector` are transparent demo components only. Replace them
with `MiniLMGuard` and an environment-scoped LLM or deterministic corrector for real use.

## TCSD scoring integration

`g2a.scoring.score_action_tokens` accepts tokenized causal-LM batches and an `action_mask`. Build
one batch with the task-matched unsafe reference and another with the safe reference, preserving
identical action token IDs. Then call:

```python
from g2a.tcsd import compute_tcsd_evidence

evidence = compute_tcsd_evidence(unsafe_token_logprobs, safe_token_logprobs)
```

Do not independently concatenate and tokenize two prompt strings: boundary tokenization can change
the action IDs and invalidate the contrast.

## Paper configuration

[`configs/paper.json`](configs/paper.json) records the reported settings:

- Qwen3-32B teacher/task backbone, frozen;
- six-layer MiniLM Guard;
- AdamW, learning rate `2e-5`, weight decay `0.01`;
- batch size 32 and 1,000 initial optimization steps;
- intervention threshold `xi=0.45`;
- at most three component-specific Skill edits;
- PPO clipping coefficient `0.2`;
- eight independent validation rollouts;
- Guard-refinement margin `0.1`.

The paper does not publish the exact editor, router, verifier prompts, reference-data schema, or
environment checkpoint API. This repository makes those boundaries explicit through protocols
instead of inventing hidden experimental details.

## Repository layout

```text
src/g2a/tcsd.py        trajectory-contrastive evidence
src/g2a/guard.py       MiniLM Guard and losses
src/g2a/scoring.py     fixed-action causal-LM scoring
src/g2a/training.py    initial and refinement training loops
src/g2a/datasets.py    TCSD and Guide-pair JSONL adapters
src/g2a/conditions.py  safe declarative Skill conditions
src/g2a/skills.py      Safety Skill representation and routing
src/g2a/runtime.py     closed-loop runtime intervention
src/g2a/policy.py      deterministic hard safety constraints
src/g2a/evolution.py   verified component-specific Skill evolution
src/g2a/audit.py       hash-chained runtime audit log
tests/                 algorithm and safety-invariant tests
```

## License

Apache-2.0.
