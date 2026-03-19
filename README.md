# 🔍 Fashion E-Commerce Return Root Cause Diagnosis

**Bayesian Network Backward Inference for Diagnosing Hidden Return Reasons**

> Course: Fundamentals of Artificial Intelligence (IN2406) · TUM  
> Author: Ata Okuzcuoglu · MSc Management & Technology (Marketing + CS)  
> Context: Personal project applying course concepts to a real marketing problem

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bayesian-marketing-attribution-model.streamlit.app)

---

## The Problem

Fashion e-commerce suffers from return rates of **25-40%**, costing the industry **$218 billion globally** (Radial, 2024). Analytics dashboards show *which* items are returned — but cannot answer **why**.

Customer-reported reasons are unreliable: surveys show only 30-40% completion, and customers frequently misreport the true reason. Meanwhile, the actual breakdown (Coresight, 2023; Rocket Returns, 2025):

| Root Cause | Share of Returns |
|---|---|
| Size / Fit Mismatch | 53–70% |
| Style / Expectation Gap | 16–23% |
| Quality / Damage | 10–13% |
| Impulse / Buyer's Regret | 8–15% |
| Intentional Bracketing | ~15% (multi-brand) |

## The Solution: Backward Inference

This tool uses a **Bayesian Network** to reason *backward* from observed signals to hidden causes — a form of **abductive inference** (inference to the best explanation):

```
Given: returned = Yes + observable order signals
Infer: P(root_cause | evidence) for 5 competing causes
Find:  Which cause is most consistent with the observed evidence?
```

**Why BN and not regression?** Regression predicts *P(returned | features)* — whether a return will happen. A BN computes *P(cause | returned=Yes, signals)* — which explanation best fits the observed pattern. Three capabilities make BN the right technique here:

1. **Backward inference** — conditioning on the outcome (`returned=Yes`) to reason about causes. This backward component contributes 2-3x more diagnostic information than the forward signal-to-cause path.
2. **Marginalization** — any combination of known/unknown signals (3¹² = 531K patterns) is handled natively. No imputation needed.
3. **Explaining away** — when one cause explains the return (e.g., bracketing via multi-size order), competing causes are automatically downweighted. This competitive reasoning requires structured graphical models.

## Network Architecture

```
OBSERVABLE (12 nodes)              ROOT CAUSES (5)              OUTCOME
═══════════════════════            ═══════════════              ═══════

size_sensitive_category ──┐
is_first_purchase ────────┼──→ 👗 SIZE_MISMATCH ─────────┐
viewed_size_guide ────────┤                               │
mobile_purchase ──────────┘                               │
                                                          │
premium_price ────────────┐                               │
is_first_purchase ────────┼──→ 📸 EXPECTATION_GAP ───────┤
mobile_purchase ──────────┘                               ├──→ RETURNED
                                                          │    (Noisy-OR)
purchased_on_discount ────┐                               │
social_media_referral ────┼──→ 💸 IMPULSE_REGRET ────────┤
mobile_purchase ──────────┤                               │
young_customer ───────────┘                               │
                                                          │
multi_size_order ─────────┐                               │
young_customer ───────────┼──→ 🔄 BRACKETING ────────────┤
high_return_history ──────┘                               │
                                                          │
slow_delivery ────────────┐                               │
multiple_items_in_order ──┼──→ 📦 QUALITY/DAMAGE ────────┘
premium_price ────────────┘
```

**18 nodes · 22 edges · All binary · 2¹⁸ = 262,144 states · Exact inference**

## Key Features

- **Root Cause Diagnosis** — Set evidence, see which of 5 causes has the highest posterior probability
- **Diagnostic Lift** — P(cause|evidence) / P(cause) reveals which cause the evidence supports most
- **What-If Simulation** — Change one signal and see how return probability shifts
- **Sensitivity Analysis** — Tornado chart showing individual signal impact on return probability; reveals Multi-Size Order as dominant driver (+50% swing)
- **Continuous Input Mode** — Enter exact values (discount %, age, price €, delivery days) instead of Yes/No — the model adjusts CPT weights via linear interpolation
- **Parameter Calibration** — Adjust all 42 parameters (priors, CPT increments, Noisy-OR strengths) via UI sliders or CSV upload to match your own platform's data
- **Academic Methodology Tab** — Full formal definition, CPT calibration, Noisy-OR model, sensitivity analysis, structural constraints
- **17 Research Citations** — Every CPT parameter grounded in industry data
- **6 Persona-Based Scenarios** — Horizontal button selector with persona names (e.g. *The Instagram Impulse Shopper*), signal detail cards, plus custom mode via sidebar toggle

## Continuous Input Mode

Real-world order signals aren't binary. A 10% discount and a 60% discount both count as "Yes" — but their effect on impulse behavior is very different.

In Custom mode, toggle **Continuous Mode** to enter exact values for 4 key signals:

| Signal | Input | Threshold | Weight Range | Effect |
|---|---|---|---|---|
| Discount | 0–80% | >0% → Yes | 0.0× – 1.8× | Modulates impulse_regret increment |
| Customer Age | 16–70 years | <40 → Yes | 0.2× – 1.5× | Modulates impulse + bracketing |
| Product Price | €5–500 | >€100 → Yes | 0.3× – 1.5× | Modulates expectation + quality |
| Delivery Time | 1–14 days | >5 days → Yes | 0.0× – 1.5× | Modulates quality/damage |

The weight modulates the CPT increment via linear interpolation between research-grounded reference points. This composes with custom parameters from the Calibration tab:

```
effective_increment = calibrated_increment × continuous_weight
```

The BN structure stays binary (18 nodes, 22 edges, same inference). Only the CPT values are modulated at build time.

## Parameter Calibration

The default parameters are grounded in 17 industry research citations — but every platform is different. A fast-fashion brand might have 38% return rates with 85% mobile traffic; a luxury brand might have 18% returns and 40% mobile.

The **Calibration** tab provides three modes:

| Mode | What You Adjust | Use Case |
|---|---|---|
| **Quick Mode** | 12 observable priors + 6 Noisy-OR strengths | "Our mobile share is 78%, not 65%" |
| **Advanced Mode** | All 24 CPT risk increments per root cause | "Size-sensitive category contributes +15%, not +12%" |
| **CSV Import/Export** | All 42 parameters via spreadsheet | Team calibration workflow, version control |

All other tabs (Diagnosis, What-If, Methodology) update live with your custom parameters. A green banner indicates when custom parameters are active.

## Example Diagnoses

| Scenario | Top Diagnosis | Lift |
|---|---|---|
| The Size-Blind First-Timer | 👗 Size Mismatch | 3.69x |
| The Try-At-Home Bracketer | 🔄 Bracketing | 12.05x |
| The Instagram Impulse Shopper | 💸 Impulse Regret | 3.62x |
| The Frustrated Late Receiver | 📦 Quality/Fulfillment | 5.56x |

## Tech Stack

- **Engine:** Custom BayesNet class with cached joint table inference — full joint computed once (2.5s), subsequent queries <20ms via NumPy boolean masking
- **Frontend:** Streamlit with 6 tabs (Diagnosis, What-If, Sensitivity, Calibration, Methodology, References)
- **No external ML libraries** — Built from first principles to demonstrate understanding
- **42 parameterized CPT values** — Extracted into a structured config with CSV import/export
- **Structural constraint:** Bracketing impossible without multi-size order (P=0.00)

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## References

See the full References tab in the app for 17 citations. Key sources:

1. Coresight Research (2023). "The True Cost of Apparel Returns"
2. Radial (2024). "Tech Takes on E-Commerce's $218 Billion Returns Problem"
3. Rocket Returns (2025). "Ecommerce Return Rates: Complete Industry Analysis"
4. AfterShip (2024). "Returns: Fashion's $218 Billion Problem"
5. Landmark Global (2025). "Wardrobing & Bracketing: Serial Returners"
6. Russell & Norvig (2021). *AI: A Modern Approach*, 4th ed., Ch. 13-14

## Part of the MarTech × AI Portfolio

This is **Project 3** in a portfolio demonstrating AI techniques applied to marketing:

| # | Project | AI Technique | Status |
|---|---------|-------------|--------|
| 1 | ContentEngine AI | LLM batch pipeline | ✅ Live |
| 2 | CSP Campaign Planner | Constraint satisfaction | ✅ Live |
| 3 | **Bayesian Return Diagnosis** | Bayesian Networks | ← You are here |
| 4 | Competitor Intel Monitor | NLP + sentiment scoring | ✅ Live |
| 5 | Journey Intelligence Engine | Markov chains + anomaly scoring | ✅ Live |
| 6 | MDP Optimal Contact Policy | Markov Decision Process | Planned |
| 7 | Logic Compliance Engine | Propositional/FOL logic | Planned |

---

*Built by Ata Okuzcuoglu · TUM MSc Management & Technology · 2026*
