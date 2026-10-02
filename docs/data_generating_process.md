# Data Generating Process

## Purpose

This document specifies the synthetic data-generating process (DGP) used in the first working paper.

The purpose of the simulator is not to reproduce a real AML environment in full detail. Instead, it provides a controlled environment in which the true latent AML state, regime, and causal mechanisms are known.

This allows us to evaluate whether uncertainty-guided investigation improves recovery of the underlying causal mechanism under a limited investigation budget.

## Unit of Analysis

The basic observational unit is an account \(i\) observed at time \(t\).

For each account-time pair, the simulator generates:

- a latent AML state;
- observable transactional and behavioural variables;
- contextual variables;
- the true causal relationships responsible for generating those observations.

The latent state and true causal graph are retained for evaluation but are not supplied to the Bayesian inference model.

## Latent AML State

Let

$$[Z_{i,t} \in \{0,1,2]$$

represent the latent AML state of account \(i\) at time \(t\).

The initial interpretation is:

\[
Z=0:\text{ normal activity}
\]

\[
Z=1:\text{ suspicious or transitional activity}
\]

\[
Z=2:\text{ active laundering mechanism}
\]

The state evolves dynamically over time.

A generic transition model is

$$[P(Z_{i,t}\mid Z_{i,t-1},R_t),]$$


where $$(R_t)$$ represents the current environmental regime.

The latent state is never directly observed by the inference model.

## Environmental Regimes

The simulator contains an environmental regime variable

$$[R_t.]$$

Regimes represent changes in the data-generating environment that may alter observable transaction behaviour.

A regime change does not necessarily imply a change in AML activity.

This distinction is important because the research design aims to separate:

- estimated AML risk;
- epistemic or posterior uncertainty;
- posterior surprise caused by observations that are inconsistent with the model's previous expectations.

Regime changes therefore provide a controlled mechanism for testing whether uncertainty and surprise contain useful information beyond estimated AML risk.

## Initial Variables

The first version of the simulator will use a deliberately small set of variables.

### Latent variable

$$[Z_{i,t}]$$

Latent AML state.

### Contextual variables

$$[E_t]$$

Environmental or economic context.

$$[N_{i,t}]$$

Network exposure of account \(i\).

### Observable variables

$$[V_{i,t}]$$

Transaction volume or transaction intensity.

$$[C_{i,t}]$$

Cross-border transaction ratio.

$$[S_{i,t}]$$

Structuring behaviour score.

$$[A_{i,t}]$$

Transaction amount behaviour.

The exact probability distributions used for these variables will be specified during implementation.

## Initial Structural Causal Model

The first simulator will implement a known structural causal model.

The conceptual graph is:

```text
Environmental context ───────────────→ Transaction volume
        │                                    ↑
        ↓                                    │
Network exposure ─────→ AML state ───────────┤
        │                    │                │
        │                    ↓                │
        └──────────→ Cross-border ratio       │
                             │                │
AML state ─────────→ Structuring behaviour ───┘
```

The graph represents the causal structure of the simulated world rather than associations estimated from the generated data.

Each endogenous variable will later be implemented as a structural equation of the form

$$[X_j := f_j(PA_j,\epsilon_j),]$$

where $$(PA_j)$$ denotes the causal parents of variable $$(X_j)$$ and $$(epsilon_j)$$ is an exogenous noise term.

## Ground Truth

For every generated dataset, the simulator retains:

1. the true causal graph;
2. the true latent AML states;
3. the environmental regime;
4. the structural parameters used to generate the observations;
5. the random seed.

These quantities constitute the experimental ground truth.

They may be used during evaluation but must not be used by the Bayesian state estimator or routing policy.

## Bayesian Observation Model

The Bayesian model observes transactional, behavioural, network, and contextual variables.

Its objective is to infer a posterior distribution over the latent AML state:

$$[P(Z_{i,t}\mid X_{i,1:t}).]$$

The posterior distribution will subsequently be used to derive quantities such as:

$$[R_{i,t}]$$

for estimated AML risk,

$$[U_{i,t}]$$

for state uncertainty, and

$$[B_{i,t}]$$

for posterior surprise.

The exact definitions of these quantities will be specified separately from the simulator.

## Investigation Budget

Only a fraction of account-time observations may be selected for local causal investigation.

For a total set of candidate cases $$(\mathcal{C})$$, an investigation policy selects a subset


$$\mathcal{I} \subseteq \mathcal{C}$$

subject to

$$[|\mathcal{I}| \leq B,]$$

where \(B\) represents the investigation budget.

Different routing policies will determine which observations enter \(\mathcal{I}\).

## Evaluation Principle

Because the simulator defines the true causal graph, causal mechanisms recovered from investigated cases can be compared directly against the ground truth.

The primary experimental comparison is therefore:

```text
known causal mechanism
        ↓
synthetic observations
        ↓
Bayesian state inference
        ↓
budget-constrained case routing
        ↓
local causal discovery
        ↓
recovered mechanism
        ↓
comparison against ground truth
```

The principal question is whether uncertainty-guided routing produces better causal mechanism recovery for the same investigation budget than alternative routing strategies.

## Scope of Version 1

The first simulator intentionally excludes several sources of complexity, including:

- multiple AML typologies;
- realistic banking data schemas;
- heterogeneous financial institutions;
- sophisticated transaction networks;
- missing data;
- latent confounding;
- measurement error;
- graph neural networks;
- learned causal mechanisms.

These may be introduced in later experiments only after the core research hypothesis can be evaluated successfully in the minimal controlled setting.