"""
Fashion E-Commerce Return Root Cause Diagnosis — Bayesian Network Engine
=========================================================================
Version: 2.1 — Continuous Input Mode + Parameter Calibration + Validation

A Bayesian Network for diagnosing the hidden root cause behind fashion
e-commerce product returns using backward (diagnostic) inference.

Academic Context:
    This project was developed as a personal project during the "Introduction
    to Artificial Intelligence" course (IN2406) at the Technical University
    of Munich (TUM), applying Bayesian Networks — specifically backward
    inference — to a real-world marketing problem.

Problem Statement:
    Fashion e-commerce suffers from return rates of 25-40%, costing the
    industry $218 billion globally (Radial, 2024). While dashboards show
    WHICH items are returned, they cannot answer WHY. Customer-reported
    return reasons are unreliable — surveys show only 30-40% completion
    rates, and customers often misreport the true reason (e.g., stating
    "didn't like it" when the actual issue was sizing).

    This Bayesian Network infers the most probable hidden root cause from
    observable order-level signals, using Bayes' theorem to reason backward
    from the observed return to competing causal explanations.

Formal Definition (Russell & Norvig, Ch. 13-14):
    A Bayesian Network is a tuple (X, G, P) where:
    - X = {X₁, ..., X₁₈}: Set of random variables
    - G = (V, E): Directed Acyclic Graph encoding conditional independence
    - P = {P(Xᵢ | Parents(Xᵢ))}: Conditional Probability Tables

    In this network:
    - 12 observable nodes (order/customer signals)
    -  5 latent root-cause nodes (hidden reasons for return)
    -  1 outcome node (returned: Yes/No)

Key Capability — Backward Inference:
    Given evidence E = {returned = Yes, observed signals...}, compute:
        P(root_cause | E) for each root cause
    Compare with prior P(root_cause) to find diagnostic lift:
        lift(C) = P(C=Yes | E) / P(C=Yes)
    The root cause with highest lift is the most probable explanation.

References:
    [1] Coresight Research (2023). "The True Cost of Apparel Returns."
        - Size/fit: 53%, Color: 16%, Damage: 10% of returns
    [2] Radial (2024). "Tech Takes on E-Commerce's $218 Billion Returns Problem."
        - Poor fit/sizing: 70% of fashion returns
    [3] Rocket Returns (2025). "Ecommerce Return Rates: Complete Industry Analysis."
        - Size/fit: 67%, Style/color: 23%, Quality: 10%
    [4] AfterShip (2024). "Returns: Fashion's $218 Billion Problem."
        - 45.9% disliked item, 21% not as described (J. Business Economics)
    [5] Landmark Global (2025). "Wardrobing & Bracketing: Serial Returners."
        - 51% of Gen Z regularly bracket, 43% wardrobing
    [6] Returnalyze (2023). "Breaking Down Bracketing."
        - ~15% of multi-brand returns due to bracketing
    [7] Statista (2022). "Top Reasons for Bracketing Online Purchases."
        - 48% bracket when sizing unclear, 36% can't try on in-store
    [8] MIT Sloan (2024). "How Better Predictive Models Could Lead to Fewer Returns."
        - Form-fitting > casual return rate; horizontal < vertical stripes
    [9] ScienceDirect (2025). "Fits Like a Glove? Size Finders and Fashion Returns."
        - Size finder users 0.65% MORE likely to return (496K orders, Sweden)
    [10] McKinsey (2021). "Returning to Order: Improving Returns Management."
        - 25% return rate apparel e-commerce; cross-functional challenge
    [11] Russell, S. & Norvig, P. "Artificial Intelligence: A Modern Approach."
        - Ch. 13: Quantifying Uncertainty; Ch. 14: Probabilistic Reasoning

Author: Ata Okuzcuoglu
Course: Fundamentals of Artificial Intelligence (IN2406), TUM
"""

import math
import numpy as np
from collections import OrderedDict
from itertools import product as iter_product


# =============================================================================
# BAYESIAN NETWORK ENGINE (Generic — reusable for any BN)
# =============================================================================

class BayesNode:
    """A node in a Bayesian Network with arbitrary discrete domain."""

    def __init__(self, name, parents, cpt, domain):
        self.name = name
        self.parents = parents if parents else []
        self.cpt = cpt          # {(parent_val_tuple,): {val: prob}}
        self.domain = domain    # list of possible values
        self.children = []

    def p(self, value, evidence):
        """P(self=value | parent values from evidence)."""
        if self.parents:
            parent_vals = tuple(evidence[p] for p in self.parents)
        else:
            parent_vals = ()
        dist = self.cpt.get(parent_vals, {})
        return dist.get(value, 0.0)

    def __repr__(self):
        return f"BayesNode({self.name}, parents={self.parents})"


class BayesNet:
    """
    Bayesian Network with exact inference via cached joint table.

    Inference: Full joint is computed once (O(n·2ⁿ)), then any query
    is answered via marginalization over the cached table (O(2ⁿ) sum).
    For 18 binary nodes: 2¹⁸ = 262,144 entries, built in ~2.5s,
    subsequent queries in <20ms.
    """

    def __init__(self):
        self.nodes = OrderedDict()
        self.variables = []
        self._joint = None       # Cached joint probability table (numpy)
        self._var_idx = None     # {variable_name: bit_index}
        self._bits_array = None  # np.arange(2**n) for masking

    def add_node(self, name, parents, cpt, domain):
        node = BayesNode(name, parents, cpt, domain)
        self.nodes[name] = node
        self.variables.append(name)
        for p in parents:
            self.nodes[p].children.append(name)
        self._joint = None  # Invalidate cache
        return node

    def joint_probability(self, assignment):
        """Compute P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ | parents(xᵢ))."""
        p = 1.0
        for var in self.variables:
            p *= self.nodes[var].p(assignment[var], assignment)
            if p == 0:
                return 0.0
        return p

    def _ensure_joint(self):
        """Build and cache the full joint probability table."""
        if self._joint is not None:
            return
        import numpy as np
        n = len(self.variables)
        self._var_idx = {v: i for i, v in enumerate(self.variables)}
        self._bits_array = np.arange(2**n)
        self._joint = np.zeros(2**n)
        for bits in range(2**n):
            assignment = {}
            for i, var in enumerate(self.variables):
                assignment[var] = "Yes" if (bits >> i) & 1 else "No"
            self._joint[bits] = self.joint_probability(assignment)

    def enumeration_ask(self, query_var, evidence):
        """
        P(query_var | evidence) via cached joint table marginalization.

        This is the core backward inference mechanism:
        Given evidence on downstream nodes (e.g., returned=Yes),
        compute posterior distribution over upstream causes.

        Implementation: builds full joint once, then masks and sums.
        First call: O(n·2ⁿ). Subsequent calls: O(2ⁿ) sum only.
        """
        self._ensure_joint()
        n = len(self.variables)

        # Build evidence mask
        mask = self._build_mask(evidence)

        p_evidence = self._joint[mask].sum()
        if p_evidence == 0:
            return {val: 0.0 for val in self.nodes[query_var].domain}

        distribution = {}
        q_bit = self._var_idx[query_var]
        for val in self.nodes[query_var].domain:
            q_match = 1 if val == "Yes" else 0
            q_mask = mask & (((self._bits_array >> q_bit) & 1) == q_match)
            distribution[val] = float(self._joint[q_mask].sum() / p_evidence)
        return distribution

    def _build_mask(self, evidence):
        """Create a boolean mask over joint table entries matching evidence."""
        import numpy as np
        n = len(self.variables)
        mask = np.ones(2**n, dtype=bool)
        for evar, eval_val in evidence.items():
            e_bit = self._var_idx[evar]
            e_match = 1 if eval_val == "Yes" else 0
            mask &= ((self._bits_array >> e_bit) & 1) == e_match
        return mask

    def recursive_enumeration_ask(self, query_var, evidence):
        """
        P(query_var | evidence) via recursive enumeration (no joint table).

        Slower for repeated queries on the same network, but faster when
        the network is rebuilt frequently (avoids 2.5s joint table build).
        Used by parameter_robustness where each variation rebuilds the network.
        """
        distribution = {}
        for val in self.nodes[query_var].domain:
            e = dict(evidence)
            e[query_var] = val
            distribution[val] = self._enumerate_all(list(self.variables), e)
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}
        return distribution

    def _enumerate_all(self, variables, evidence):
        if not variables:
            return 1.0
        first = variables[0]
        rest = variables[1:]
        if first in evidence:
            return (self.nodes[first].p(evidence[first], evidence)
                    * self._enumerate_all(rest, evidence))
        else:
            total = 0.0
            for val in self.nodes[first].domain:
                e = dict(evidence)
                e[first] = val
                total += (self.nodes[first].p(val, e)
                          * self._enumerate_all(rest, e))
            return total


# =============================================================================
# NETWORK DEFINITION: Fashion Return Root Cause Diagnosis
# =============================================================================

# Node metadata for UI display and academic documentation
NODE_META = OrderedDict({
    # --- LAYER 1: Observable Order/Customer Signals ---
    "size_sensitive_category": {
        "label": "Size-Sensitive Category",
        "description": "Whether the product category has high fit sensitivity",
        "examples": "Yes = Dresses, Pants, Shoes, Outerwear; No = T-shirts, Accessories, Bags",
        "layer": "Product Attributes",
        "citation": "[3] Rocket Returns: Dresses/pants return rates 25-35%, accessories ~12%",
    },
    "is_first_purchase": {
        "label": "First Purchase (Brand-New Customer)",
        "description": "Customer has no prior order history with this brand",
        "examples": "Yes = First time buyer; No = Returning customer",
        "layer": "Customer Profile",
        "citation": "[10] McKinsey: First-time online shoppers show 67% higher return rates",
    },
    "viewed_size_guide": {
        "label": "Viewed Size Guide",
        "description": "Customer clicked on/viewed the size guide before purchasing",
        "examples": "Yes = Viewed; No = Did not view",
        "layer": "Browsing Behavior",
        "citation": "[9] ScienceDirect (2025): Size finder users 0.65% MORE likely to return (n=496,365)",
    },
    "mobile_purchase": {
        "label": "Mobile Purchase",
        "description": "Order placed from a mobile device (smaller screen, less detail review)",
        "examples": "Yes = Mobile/Tablet; No = Desktop",
        "layer": "Browsing Behavior",
        "citation": "[3] Rocket Returns: Mobile purchases show elevated return rates vs desktop",
    },
    "premium_price": {
        "label": "Premium/Luxury Price",
        "description": "Product is in the premium or luxury price segment",
        "examples": "Yes = Premium/Luxury (€100+); No = Budget/Mid-range",
        "layer": "Product Attributes",
        "citation": "[19] Return stats: Luxury 15-20% return rate vs fast fashion 38%",
    },
    "purchased_on_discount": {
        "label": "Purchased on Discount",
        "description": "Item was purchased during a sale or with a discount code",
        "examples": "Yes = Discounted; No = Full price",
        "layer": "Purchase Context",
        "citation": "[23] Radial: Emotionally driven sales + impulse → higher returns",
    },
    "social_media_referral": {
        "label": "Social Media Referral",
        "description": "Customer came from Instagram, TikTok, or other social platform",
        "examples": "Yes = Social media traffic; No = Direct/search/email",
        "layer": "Purchase Context",
        "citation": "[3] Rocket Returns: Social media purchases show HIGHEST return rates",
    },
    "young_customer": {
        "label": "Young Customer (Gen Z / Millennial)",
        "description": "Customer is under 40 (Gen Z or Millennial demographic)",
        "examples": "Yes = Under 40; No = 40+",
        "layer": "Customer Profile",
        "citation": "[5] Landmark Global: 51% of Gen Z regularly bracket purchases",
    },
    "multi_size_order": {
        "label": "Multi-Size Order (Bracketing Signal)",
        "description": "Customer ordered 2+ sizes of the same product in one order",
        "examples": "Yes = Same product, multiple sizes; No = Single size",
        "layer": "Purchase Context",
        "citation": "[6] Returnalyze: ~15% of multi-brand returns due to bracketing; [7] Statista: 48% bracket when sizing unclear",
    },
    "high_return_history": {
        "label": "High Return History",
        "description": "Customer has above-average historical return rate (>30%)",
        "examples": "Yes = Serial returner (>30% past returns); No = Normal return rate",
        "layer": "Customer Profile",
        "citation": "[15] ZigZag: AI models assign customer return 'scores'; ASOS flags/limits high-return accounts",
    },
    "slow_delivery": {
        "label": "Slow Delivery (>5 days)",
        "description": "Order took longer than 5 business days to arrive",
        "examples": "Yes = >5 days; No = ≤5 days",
        "layer": "Fulfillment",
        "citation": "[24] Opensend: Damaged items during shipping ~20% of returns",
    },
    "multiple_items_in_order": {
        "label": "Multiple Items in Order",
        "description": "Order contains more than one distinct product",
        "examples": "Yes = 2+ different products; No = Single item",
        "layer": "Purchase Context",
        "citation": "[20] MDPI (2024): Number of products in order is predictive feature for returns",
    },

    # --- LAYER 2: Hidden Root Causes (Latent — inferred by BN) ---
    "size_mismatch": {
        "label": "👗 Size/Fit Mismatch",
        "description": "Product did not fit as expected — wrong size, unexpected cut, brand sizing inconsistency",
        "layer": "Root Cause",
        "citation": "[1] Coresight: 53% of returns; [2] Radial: 70% of fashion returns; [3] Rocket Returns: 67%",
        "base_rate": "~55% of all fashion returns",
    },
    "expectation_gap": {
        "label": "📸 Expectation Gap",
        "description": "Product looks/feels different than online presentation — color, fabric, quality mismatch",
        "layer": "Root Cause",
        "citation": "[4] AfterShip/J.Bus.Econ: 21% 'not as described'; [1] Coresight: 16% color issues",
        "base_rate": "~20% of all fashion returns",
    },
    "impulse_regret": {
        "label": "💸 Impulse/Buyer's Regret",
        "description": "Customer made an emotional/impulsive purchase and later reconsidered",
        "layer": "Root Cause",
        "citation": "[23] Radial: buyer's remorse significant in fashion; [24] Opensend: 15-20% of returns",
        "base_rate": "~12% of all fashion returns",
    },
    "bracketing": {
        "label": "🔄 Intentional Bracketing",
        "description": "Customer deliberately ordered multiple sizes/variants to try at home",
        "layer": "Root Cause",
        "citation": "[5] Landmark Global: 51% Gen Z brackets; [6] Returnalyze: ~15% of returns; [7] Statista: 48% bracket when sizing unclear",
        "base_rate": "~8% of all fashion returns",
    },
    "quality_or_damage": {
        "label": "📦 Quality / Fulfillment Issue",
        "description": "Product arrived damaged, defective, or with quality below expectations",
        "layer": "Root Cause",
        "citation": "[1] Coresight: Damage 10%; [2] Radial: 13% faulty goods; [24] Opensend: ~20% damaged in shipping",
        "base_rate": "~10% of all fashion returns",
    },

    # --- LAYER 3: Outcome ---
    "returned": {
        "label": "🔙 Product Returned",
        "description": "Customer initiated a return for this order",
        "layer": "Outcome",
        "citation": "[10] McKinsey: 25% e-commerce apparel return rate; [2] Radial: 30% average online clothing",
        "base_rate": "~28% overall fashion e-commerce return rate",
    },
})

# Observable nodes (user can set evidence on these)
OBSERVABLE_NODES = [
    "size_sensitive_category", "is_first_purchase", "viewed_size_guide",
    "mobile_purchase", "premium_price", "purchased_on_discount",
    "social_media_referral", "young_customer", "multi_size_order",
    "high_return_history", "slow_delivery", "multiple_items_in_order",
]

# Root cause nodes (these are what we diagnose)
ROOT_CAUSE_NODES = [
    "size_mismatch", "expectation_gap", "impulse_regret",
    "bracketing", "quality_or_damage",
]

# Group observable nodes by layer for UI
OBSERVABLE_GROUPS = OrderedDict({
    "🏷️ Product Attributes": ["size_sensitive_category", "premium_price"],
    "👤 Customer Profile": ["is_first_purchase", "young_customer", "high_return_history"],
    "🖱️ Browsing Behavior": ["viewed_size_guide", "mobile_purchase"],
    "🛒 Purchase Context": ["purchased_on_discount", "social_media_referral",
                            "multi_size_order", "multiple_items_in_order"],
    "📦 Fulfillment": ["slow_delivery"],
})


# =============================================================================
# CONTINUOUS INPUT PROFILES
# =============================================================================
# These define how real-world continuous values (discount %, age, price €, days)
# map to intensity weights that modulate the binary CPT increments.
#
# Flow:  continuous value → compute_weight() → weight
#        calibrated increment × weight = effective increment
#
# Reference points are grounded in industry research:
# - Discount: Radial [23] shows impulse returns scale with discount depth
# - Age: Landmark Global [5] shows Gen Z (18-24) brackets 51% vs 25% for 35+
# - Price: Return behavior differs significantly by price segment [19]
# - Delivery: Opensend [24] shows damage increases with transit time

CONTINUOUS_INPUTS = OrderedDict({
    "purchased_on_discount": {
        "label": "Discount Amount",
        "unit": "%",
        "min_val": 0,
        "max_val": 80,
        "default_val": 0,
        "step": 5,
        "threshold": 0,          # > threshold → Yes
        "direction": "above",    # "above" = higher value → Yes
        "help": "0% = full price. Higher discount → stronger impulse signal.",
        "ref_points": [
            (0, 0.0),      # No discount → no effect
            (10, 0.5),     # Small discount → mild impulse signal
            (25, 1.0),     # Moderate discount → baseline (matches default CPT)
            (50, 1.5),     # Deep discount → strong impulse signal [23]
            (80, 1.8),     # Extreme clearance → very strong
        ],
        # Which cause increments this continuous input modulates
        "affects": {
            "impulse_regret": ["purchased_on_discount"],
        },
    },
    "young_customer": {
        "label": "Customer Age",
        "unit": "years",
        "min_val": 16,
        "max_val": 70,
        "default_val": 30,
        "step": 1,
        "threshold": 40,          # < threshold → Yes (inverted)
        "direction": "below",     # "below" = lower value → Yes
        "help": "Under 40 = Yes. Younger customers bracket and impulse-buy more [5].",
        "ref_points": [
            (18, 1.5),     # Gen Z peak — 51% bracket [5]
            (25, 1.2),     # Late Gen Z / early Millennial
            (35, 1.0),     # Baseline (matches default CPT)
            (45, 0.5),     # Gen X — lower impulse/bracketing
            (60, 0.2),     # Boomers — minimal bracketing
        ],
        "affects": {
            "impulse_regret": ["young_customer"],
            "bracketing": ["young_customer_with", "young_customer_without"],
        },
    },
    "premium_price": {
        "label": "Product Price",
        "unit": "€",
        "min_val": 5,
        "max_val": 500,
        "default_val": 120,
        "step": 5,
        "threshold": 100,         # > threshold → Yes
        "direction": "above",
        "help": "Above €100 = premium. Higher price → stronger expectation gap signal [19].",
        "ref_points": [
            (10, 0.3),     # Budget items — low return risk
            (50, 0.6),     # Mid-range
            (100, 1.0),    # Premium threshold — baseline
            (250, 1.3),    # Luxury — higher expectation
            (500, 1.5),    # Ultra-luxury — maximum scrutiny
        ],
        "affects": {
            "expectation_gap": ["premium_price"],
            "quality_or_damage": ["premium_price"],
        },
    },
    "slow_delivery": {
        "label": "Delivery Time",
        "unit": "days",
        "min_val": 1,
        "max_val": 14,
        "default_val": 3,
        "step": 1,
        "threshold": 5,           # > threshold → Yes
        "direction": "above",
        "help": "Over 5 days = slow. Longer transit → more damage/quality risk [24].",
        "ref_points": [
            (1, 0.0),      # Next-day — no delay effect
            (3, 0.3),      # Standard fast
            (5, 1.0),      # Baseline slow threshold
            (8, 1.3),      # Notably slow
            (14, 1.5),     # Very slow — maximum damage risk
        ],
        "affects": {
            "quality_or_damage": ["slow_delivery"],
        },
    },
})


def compute_weight(value, ref_points):
    """
    Linear interpolation between reference points.

    Parameters
    ----------
    value : float
        The continuous input value (e.g., discount=45, age=22, price=200).
    ref_points : list of (value, weight) tuples
        Sorted reference points defining the interpolation curve.

    Returns
    -------
    float
        Interpolated weight, clamped to the range of the reference points.

    Example
    -------
    >>> compute_weight(37, [(10, 0.5), (25, 1.0), (50, 1.5)])
    1.24  # linearly interpolated between (25, 1.0) and (50, 1.5)
    """
    if value <= ref_points[0][0]:
        return ref_points[0][1]
    if value >= ref_points[-1][0]:
        return ref_points[-1][1]
    for i in range(len(ref_points) - 1):
        v0, w0 = ref_points[i]
        v1, w1 = ref_points[i + 1]
        if v0 <= value <= v1:
            t = (value - v0) / (v1 - v0)
            return w0 + t * (w1 - w0)
    return 1.0  # fallback (should never reach here)


def continuous_to_evidence_and_weights(continuous_values):
    """
    Convert continuous inputs to binary evidence + intensity weights.

    Parameters
    ----------
    continuous_values : dict
        Maps node name → continuous value. E.g., {"purchased_on_discount": 45, "young_customer": 22}
        Only nodes present in CONTINUOUS_INPUTS are processed; others are ignored.

    Returns
    -------
    evidence_updates : dict
        Maps node name → "Yes"/"No" based on threshold.
    weights : dict
        Maps node name → interpolated weight for increment modulation.

    Example
    -------
    >>> continuous_to_evidence_and_weights({"purchased_on_discount": 45})
    ({"purchased_on_discount": "Yes"}, {"purchased_on_discount": 1.4})
    """
    evidence_updates = {}
    weights = {}

    for node, value in continuous_values.items():
        if node not in CONTINUOUS_INPUTS:
            continue
        config = CONTINUOUS_INPUTS[node]

        # Determine Yes/No from threshold
        if config["direction"] == "above":
            evidence_updates[node] = "Yes" if value > config["threshold"] else "No"
        else:  # "below"
            evidence_updates[node] = "Yes" if value < config["threshold"] else "No"

        # Compute intensity weight
        weights[node] = compute_weight(value, config["ref_points"])

    return evidence_updates, weights


def apply_intensity_weights(params, weights):
    """
    Modulate CPT increments by intensity weights from continuous inputs.

    This composes with custom parameters from the Calibration tab:
        effective_increment = calibrated_increment × weight

    Parameters
    ----------
    params : dict
        Full params dict (from get_params, may include custom calibration).
    weights : dict
        Maps observable node name → weight multiplier.
        E.g., {"purchased_on_discount": 1.5, "young_customer": 1.2}

    Returns
    -------
    dict
        New params dict with modulated increments. Original is not modified.
    """
    import copy
    modified = copy.deepcopy(params)

    for node, weight in weights.items():
        if node not in CONTINUOUS_INPUTS:
            continue
        config = CONTINUOUS_INPUTS[node]

        for cause, inc_keys in config["affects"].items():
            if cause not in modified["increments"]:
                continue
            for key in inc_keys:
                if key in modified["increments"][cause]:
                    original = modified["increments"][cause][key]
                    modified["increments"][cause][key] = min(
                        round(original * weight, 4), 0.95
                    )

    return modified


# =============================================================================
# PARAMETERIZED CONFIGURATION
# =============================================================================
# All CPT parameters extracted into a single config dict.
# Users can override any subset — unspecified values fall back to defaults.
# This enables Improvement #1: "Does this work with my own data?"
#
# Structure:
#   priors        — P(observable=Yes) for each root node (no parents)
#   increments    — Additive risk contributions for each root cause CPT
#   outcome       — Noisy-OR leak probability + per-cause strengths
#
# Notation: {(parent_val_tuple): {value: probability}}
# For root nodes: {(): {value: probability}}

DEFAULT_PARAMS = {
    # ---------------------------------------------------------------
    # Observable priors: P(node = Yes)
    # ---------------------------------------------------------------
    "priors": {
        "size_sensitive_category": 0.55,   # [3] ~55% orders in fit-sensitive categories
        "is_first_purchase":      0.35,   # [10] ~35% first-time buyers
        "viewed_size_guide":      0.25,   # [9] ~25% view size guide
        "mobile_purchase":        0.65,   # [3] ~65% mobile traffic
        "premium_price":          0.20,   # [19] ~20% premium/luxury segment
        "purchased_on_discount":  0.40,   # ~40% orders involve discount
        "social_media_referral":  0.25,   # [3] ~25% social media traffic
        "young_customer":         0.55,   # [5] ~55% Gen Z + Millennial
        "multi_size_order":       0.08,   # [6] ~8% bracket with multiple sizes
        "high_return_history":    0.15,   # [15] ~15% serial returners
        "slow_delivery":          0.20,   # ~20% orders >5 day delivery
        "multiple_items_in_order": 0.35,  # ~35% multi-item orders
    },

    # ---------------------------------------------------------------
    # CPT increments: additive risk model for each root cause
    # base + sum(increment_i if parent_i=Yes)
    # ---------------------------------------------------------------
    "increments": {
        "size_mismatch": {
            "base": 0.04,
            "size_sensitive_category": 0.12,  # [3] fit-sensitive categories
            "is_first_purchase":      0.08,  # [10] no brand sizing knowledge
            "viewed_size_guide":      0.03,  # [9] signals uncertainty
            "mobile_purchase":        0.03,  # [3] small screen less detail
            "premium_price":          0.04,  # [19] luxury brands have inconsistent sizing across collections
        },
        "expectation_gap": {
            "base": 0.03,
            "premium_price":          0.06,  # higher expectations
            "is_first_purchase":      0.05,  # no brand calibration
            "mobile_purchase":        0.03,  # screen color/detail limits
            "social_media_referral":  0.03,  # [3] filtered photos distort product appearance
        },
        "impulse_regret": {
            "base": 0.02,
            "purchased_on_discount":  0.04,  # discount triggers impulse
            "social_media_referral":  0.05,  # [3] social = highest return
            "mobile_purchase":        0.02,  # easier impulse on mobile
            "young_customer":         0.03,  # [5] Gen Z/Millennial behavior
        },
        "bracketing": {
            # Bracketing requires multi-size order — without it, probability is 0
            "base_with_multi_size":    0.70,  # very strong bracketing signal
            "base_without_multi_size": 0.00,  # impossible without multi-size
            "young_customer_with":     0.10,  # [5] Gen Z brackets more
            "young_customer_without":  0.00,  # no multi-size = no bracketing
            "high_return_history_with":  0.08,  # serial returners bracket
            "high_return_history_without": 0.00,  # no multi-size = no bracketing
            "size_sensitive_category_with": 0.05,  # [7] 48% bracket when sizing unclear; fit-sensitive = unclear
            "size_sensitive_category_without": 0.00,  # no multi-size = no bracketing
        },
        "quality_or_damage": {
            "base": 0.03,
            "slow_delivery":           0.05,  # delayed = more handling
            "multiple_items_in_order": 0.02,  # complex packaging
            "premium_price":           0.03,  # higher quality bar
        },
    },

    # ---------------------------------------------------------------
    # Noisy-OR outcome model
    # P(return=No) = (1 - leak) × ∏(1 - strength_i) for active causes
    # ---------------------------------------------------------------
    "outcome": {
        "leak": 0.05,  # returns with no modeled cause (gifts, abuse, etc.)
        "strengths": {
            "size_mismatch":    0.80,  # very likely to return if doesn't fit
            "expectation_gap":  0.65,  # often return if looks different
            "impulse_regret":   0.50,  # 50/50 — some keep despite regret
            "bracketing":       0.92,  # almost always return extra sizes
            "quality_or_damage": 0.85, # very likely to return defective
        },
    },
}


def _deep_merge(base, override):
    """Recursively merge override into base dict, returning new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def get_params(custom_params=None):
    """
    Return the full parameter set, merging any custom overrides with defaults.

    Parameters
    ----------
    custom_params : dict or None
        Partial override dict matching DEFAULT_PARAMS structure.
        Any unspecified keys fall back to defaults.

    Returns
    -------
    dict
        Complete parameter set.
    """
    if custom_params is None:
        import copy
        return copy.deepcopy(DEFAULT_PARAMS)
    return _deep_merge(DEFAULT_PARAMS, custom_params)


def params_to_flat_csv_rows(params=None):
    """
    Export parameters as flat list of (section, parameter, value) tuples.
    Suitable for CSV export.
    """
    if params is None:
        params = DEFAULT_PARAMS
    rows = []
    # Priors
    for node, val in params["priors"].items():
        rows.append(("priors", node, val))
    # Increments
    for cause, inc_dict in params["increments"].items():
        for param, val in inc_dict.items():
            rows.append(("increments", f"{cause}.{param}", val))
    # Outcome
    rows.append(("outcome", "leak", params["outcome"]["leak"]))
    for cause, val in params["outcome"]["strengths"].items():
        rows.append(("outcome", f"strength.{cause}", val))
    return rows


def flat_csv_rows_to_params(rows):
    """
    Parse flat CSV rows [(section, parameter, value), ...] into params dict.
    Inverse of params_to_flat_csv_rows.

    Validates:
    - Row count limit (max 200 rows)
    - Section names must be in {priors, increments, outcome}
    - Parameter names must be whitelisted (from DEFAULT_PARAMS)
    - Values must be finite numbers in [0.0, 1.0]
    """
    MAX_ROWS = 200

    if len(rows) > MAX_ROWS:
        raise ValueError(f"CSV exceeds maximum row limit ({MAX_ROWS}). Got {len(rows)} rows.")

    # Build whitelist from DEFAULT_PARAMS
    valid_keys = set()
    for node in DEFAULT_PARAMS["priors"]:
        valid_keys.add(("priors", node))
    for cause, inc_dict in DEFAULT_PARAMS["increments"].items():
        for param in inc_dict:
            valid_keys.add(("increments", f"{cause}.{param}"))
    valid_keys.add(("outcome", "leak"))
    for cause in DEFAULT_PARAMS["outcome"]["strengths"]:
        valid_keys.add(("outcome", f"strength.{cause}"))

    params = {"priors": {}, "increments": {}, "outcome": {"strengths": {}}}
    warnings = []

    # Skip header row if present (user may have added column headers in spreadsheet)
    if rows and rows[0][0].lower() in ("section", "type", "category"):
        rows = rows[1:]

    for section, param, value in rows:
        # Validate section
        if section not in ("priors", "increments", "outcome"):
            warnings.append(f"Unknown section '{section}' — skipped")
            continue

        # Validate key exists in whitelist
        if (section, param) not in valid_keys:
            warnings.append(f"Unknown parameter '{section}.{param}' — skipped")
            continue

        # Validate value
        val = float(value)
        if not _is_valid_probability(val):
            warnings.append(f"Invalid value {val} for '{param}' — must be finite number in [0, 1]. Skipped.")
            continue

        if section == "priors":
            params["priors"][param] = val
        elif section == "increments":
            cause, key = param.split(".", 1)
            if cause not in params["increments"]:
                params["increments"][cause] = {}
            params["increments"][cause][key] = val
        elif section == "outcome":
            if param == "leak":
                params["outcome"]["leak"] = val
            elif param.startswith("strength."):
                cause = param.split(".", 1)[1]
                params["outcome"]["strengths"][cause] = val

    if warnings:
        params["_warnings"] = warnings

    return params


def _is_valid_probability(val):
    """Check if a value is a valid probability: finite float in [0.0, 1.0]."""
    import math
    if not isinstance(val, (int, float)):
        return False
    if math.isnan(val) or math.isinf(val):
        return False
    if val < 0.0 or val > 1.0:
        return False
    return True


def validate_params(params):
    """
    Validate a full params dict. Returns (is_valid, errors) tuple.

    Checks:
    - All priors in [0, 1]
    - All increments in [0, 1]
    - Noisy-OR strengths in [0, 1]
    - Leak in [0, 1]
    - No inf/nan values anywhere
    """
    errors = []

    if "priors" in params:
        for node, val in params["priors"].items():
            if not _is_valid_probability(val):
                errors.append(f"Prior '{node}' = {val} — must be in [0, 1]")

    if "increments" in params:
        for cause, inc_dict in params["increments"].items():
            for param, val in inc_dict.items():
                if not _is_valid_probability(val):
                    errors.append(f"Increment '{cause}.{param}' = {val} — must be in [0, 1]")

    if "outcome" in params:
        if "leak" in params["outcome"]:
            if not _is_valid_probability(params["outcome"]["leak"]):
                errors.append(f"Leak = {params['outcome']['leak']} — must be in [0, 1]")
        if "strengths" in params["outcome"]:
            for cause, val in params["outcome"]["strengths"].items():
                if not _is_valid_probability(val):
                    errors.append(f"Strength '{cause}' = {val} — must be in [0, 1]")

    # Cross-parameter validation: check that base + all increments < 0.95
    # (prevents silent clamping that removes discriminative power)
    if "increments" in params:
        for cause, inc_dict in params["increments"].items():
            # For bracketing, use the "with" keys for sum check
            if cause == "bracketing":
                max_sum = inc_dict.get("base_with_multi_size", 0)
                for k, v in inc_dict.items():
                    if k.endswith("_with") and k != "base_with_multi_size":
                        max_sum += v
            else:
                max_sum = sum(v for v in inc_dict.values() if isinstance(v, (int, float)))
            if max_sum > 0.95:
                errors.append(
                    f"Increments for '{cause}' sum to {max_sum:.3f} — "
                    f"all CPT entries will be clamped to 0.95, losing discriminative power. "
                    f"Reduce increments so the maximum sum stays below 0.95."
                )

    return (len(errors) == 0, errors)


# =============================================================================
# INDUSTRY PRESETS
# =============================================================================
# Pre-configured parameter sets for common fashion e-commerce segments.
# Users pick a preset, then fine-tune with their own metrics.

INDUSTRY_PRESETS = OrderedDict({
    "fast_fashion": {
        "label": "Fast Fashion",
        "examples": "Zara, H&M, ASOS, Boohoo, SHEIN",
        "typical_return_rate": "30-40%",
        "priors": {
            "premium_price": 0.08, "purchased_on_discount": 0.55,
            "young_customer": 0.65, "mobile_purchase": 0.78,
            "social_media_referral": 0.35, "multi_size_order": 0.12,
            "is_first_purchase": 0.40, "size_sensitive_category": 0.55,
            "high_return_history": 0.20, "slow_delivery": 0.15,
            "viewed_size_guide": 0.20, "multiple_items_in_order": 0.40,
        },
    },
    "premium_luxury": {
        "label": "Premium / Luxury",
        "examples": "Mytheresa, NET-A-PORTER, SSENSE, MatchesFashion",
        "typical_return_rate": "15-25%",
        "priors": {
            "premium_price": 0.70, "purchased_on_discount": 0.20,
            "young_customer": 0.35, "mobile_purchase": 0.45,
            "social_media_referral": 0.15, "multi_size_order": 0.04,
            "is_first_purchase": 0.25, "size_sensitive_category": 0.60,
            "high_return_history": 0.08, "slow_delivery": 0.20,
            "viewed_size_guide": 0.35, "multiple_items_in_order": 0.25,
        },
    },
    "mid_range_multi_brand": {
        "label": "Mid-Range Multi-Brand",
        "examples": "Zalando, FARFETCH, About You, Nordstrom",
        "typical_return_rate": "25-35%",
        "priors": {
            "premium_price": 0.25, "purchased_on_discount": 0.40,
            "young_customer": 0.55, "mobile_purchase": 0.65,
            "social_media_referral": 0.25, "multi_size_order": 0.08,
            "is_first_purchase": 0.35, "size_sensitive_category": 0.55,
            "high_return_history": 0.15, "slow_delivery": 0.20,
            "viewed_size_guide": 0.25, "multiple_items_in_order": 0.35,
        },
    },
})


def auto_calibrate_to_return_rate(target_return_rate, custom_params=None, tolerance=0.005):
    """
    Automatically calibrate Noisy-OR parameters to hit a target return rate.

    Two-stage approach:
    1. Adjust leak probability (fast, direct impact on baseline return rate)
    2. If leak alone can't reach target, scale all Noisy-OR strengths uniformly

    Parameters
    ----------
    target_return_rate : float
        Desired P(returned=Yes), e.g. 0.38 for 38%.
    custom_params : dict or None
        Base parameters to calibrate from (e.g., an industry preset).
    tolerance : float
        Acceptable deviation from target (default: 0.5%).

    Returns
    -------
    dict with keys:
        - params: Full calibrated parameter dict (ready for build_return_diagnosis_network)
        - leak: Calibrated leak value
        - strength_scale: Multiplier applied to Noisy-OR strengths (1.0 if leak alone sufficed)
        - achieved: Actual P(returned=Yes) of calibrated network
        - method: "leak_only" or "leak_and_scale"
    """
    import copy
    params = get_params(custom_params)

    # Stage 1: Binary search on leak
    lo, hi = 0.005, 0.50
    best_leak, best_mr = params["outcome"]["leak"], 0.33
    for _ in range(20):
        mid = (lo + hi) / 2
        tp = copy.deepcopy(params)
        tp["outcome"]["leak"] = mid
        test_bn = build_return_diagnosis_network(tp)
        mr = compute_marginal_return_rate(test_bn)
        best_leak, best_mr = mid, mr
        if abs(mr - target_return_rate) < tolerance:
            tp["outcome"]["leak"] = round(mid, 4)
            return {
                "params": tp, "leak": round(mid, 4),
                "strength_scale": 1.0, "achieved": round(mr, 4),
                "method": "leak_only",
            }
        elif mr < target_return_rate:
            lo = mid
        else:
            hi = mid

    # Stage 2: Leak alone wasn't enough → scale strengths
    if best_mr > target_return_rate:
        tp2 = copy.deepcopy(params)
        tp2["outcome"]["leak"] = 0.005
        lo_s, hi_s = 0.3, 1.0
        for _ in range(20):
            mid_s = (lo_s + hi_s) / 2
            tp_s = copy.deepcopy(tp2)
            for cause in tp_s["outcome"]["strengths"]:
                tp_s["outcome"]["strengths"][cause] = round(
                    params["outcome"]["strengths"][cause] * mid_s, 4
                )
            test_bn = build_return_diagnosis_network(tp_s)
            mr = compute_marginal_return_rate(test_bn)
            if abs(mr - target_return_rate) < tolerance:
                return {
                    "params": tp_s, "leak": 0.005,
                    "strength_scale": round(mid_s, 3), "achieved": round(mr, 4),
                    "method": "leak_and_scale",
                }
            elif mr < target_return_rate:
                lo_s = mid_s
            else:
                hi_s = mid_s
        return {
            "params": tp_s, "leak": 0.005,
            "strength_scale": round(mid_s, 3), "achieved": round(mr, 4),
            "method": "leak_and_scale",
        }

    # Leak was already best we could do
    tp_final = copy.deepcopy(params)
    tp_final["outcome"]["leak"] = round(best_leak, 4)
    return {
        "params": tp_final, "leak": round(best_leak, 4),
        "strength_scale": 1.0, "achieved": round(best_mr, 4),
        "method": "leak_only",
    }


def build_return_diagnosis_network(custom_params=None):
    """
    Construct the Fashion Return Root Cause Diagnosis Bayesian Network.

    Parameters
    ----------
    custom_params : dict or None
        Optional parameter overrides. Any subset of DEFAULT_PARAMS structure.
        Unspecified values fall back to research-grounded defaults.
        See DEFAULT_PARAMS for the full specification.

    Network topology (18 nodes, 22 edges):

    OBSERVABLE SIGNALS          ROOT CAUSES              OUTCOME
    ═══════════════════         ═══════════              ═══════

    size_sensitive_category ─┐
    is_first_purchase ───────┼─→ size_mismatch ─────────┐
    viewed_size_guide ───────┤                           │
    mobile_purchase ─────────┘                           │
                                                         │
    premium_price ───────────┐                           │
    is_first_purchase ───────┼─→ expectation_gap ────────┤
    mobile_purchase ─────────┘                           │
                                                         ├─→ returned
    purchased_on_discount ───┐                           │
    social_media_referral ───┼─→ impulse_regret ─────────┤
    mobile_purchase ─────────┤                           │
    young_customer ──────────┘                           │
                                                         │
    multi_size_order ────────┐                           │
    young_customer ──────────┼─→ bracketing ─────────────┤
    high_return_history ─────┘                           │
                                                         │
    slow_delivery ───────────┐                           │
    multiple_items_in_order ─┼─→ quality_or_damage ──────┘
    premium_price ───────────┘

    Returns
    -------
    BayesNet
        Fully specified Bayesian Network ready for inference.
    """
    bn = BayesNet()
    YN = ["Yes", "No"]  # Standard binary domain

    # Merge custom params with defaults
    params = get_params(custom_params)
    priors = params["priors"]
    increments = params["increments"]
    outcome = params["outcome"]

    # -----------------------------------------------------------------
    # LAYER 1: Observable root nodes (unconditional priors)
    # -----------------------------------------------------------------
    # These represent the marginal distribution of each signal across
    # a typical fashion e-commerce order population.

    for node in OBSERVABLE_NODES:
        p_yes = priors[node]
        bn.add_node(node, [], {
            (): {"Yes": p_yes, "No": round(1 - p_yes, 4)}
        }, YN)

    # -----------------------------------------------------------------
    # LAYER 2: Root Cause nodes (conditional on observable parents)
    # -----------------------------------------------------------------

    # SIZE MISMATCH — P(size_mismatch | size_sensitive, first_purchase, viewed_guide, mobile, premium)
    # [1][2][3] Size/fit is #1 return reason: 53-70% of fashion returns
    # [19] Luxury brands have inconsistent sizing across collections
    size_parents = ["size_sensitive_category", "is_first_purchase",
                    "viewed_size_guide", "mobile_purchase", "premium_price"]
    size_inc = increments["size_mismatch"]
    size_cpt = {}
    for cat, first, guide, mobile, premium in iter_product(YN, repeat=5):
        base = size_inc["base"]
        if cat == "Yes":     base += size_inc["size_sensitive_category"]
        if first == "Yes":   base += size_inc["is_first_purchase"]
        if guide == "Yes":   base += size_inc["viewed_size_guide"]
        if mobile == "Yes":  base += size_inc["mobile_purchase"]
        if premium == "Yes": base += size_inc["premium_price"]
        base = min(base, 0.95)
        size_cpt[(cat, first, guide, mobile, premium)] = {"Yes": round(base, 3),
                                                           "No": round(1 - base, 3)}
    bn.add_node("size_mismatch", size_parents, size_cpt, YN)

    # EXPECTATION GAP — P(expectation_gap | premium_price, first_purchase, mobile, social_media)
    # [4] 21% "not as described" + [1] 16% color issues
    # [3] Social media filtered photos distort product appearance
    exp_parents = ["premium_price", "is_first_purchase", "mobile_purchase",
                   "social_media_referral"]
    exp_inc = increments["expectation_gap"]
    exp_cpt = {}
    for price, first, mobile, social in iter_product(YN, repeat=4):
        base = exp_inc["base"]
        if price == "Yes":  base += exp_inc["premium_price"]
        if first == "Yes":  base += exp_inc["is_first_purchase"]
        if mobile == "Yes": base += exp_inc["mobile_purchase"]
        if social == "Yes": base += exp_inc["social_media_referral"]
        base = min(base, 0.95)
        exp_cpt[(price, first, mobile, social)] = {"Yes": round(base, 3),
                                                    "No": round(1 - base, 3)}
    bn.add_node("expectation_gap", exp_parents, exp_cpt, YN)

    # IMPULSE REGRET — P(impulse_regret | discount, social_media, mobile, young)
    # [23] Radial: Emotionally driven + buyer's remorse significant
    imp_parents = ["purchased_on_discount", "social_media_referral",
                   "mobile_purchase", "young_customer"]
    imp_inc = increments["impulse_regret"]
    imp_cpt = {}
    for disc, social, mobile, young in iter_product(YN, repeat=4):
        base = imp_inc["base"]
        if disc == "Yes":    base += imp_inc["purchased_on_discount"]
        if social == "Yes":  base += imp_inc["social_media_referral"]
        if mobile == "Yes":  base += imp_inc["mobile_purchase"]
        if young == "Yes":   base += imp_inc["young_customer"]
        base = min(base, 0.95)
        imp_cpt[(disc, social, mobile, young)] = {"Yes": round(base, 3),
                                                   "No": round(1 - base, 3)}
    bn.add_node("impulse_regret", imp_parents, imp_cpt, YN)

    # BRACKETING — P(bracketing | multi_size_order, young, high_return_history, size_sensitive)
    # [5] 51% of Gen Z bracket; [6] ~15% of returns from bracketing
    # [7] 48% bracket when sizing unclear — fit-sensitive categories = unclear sizing
    brack_parents = ["multi_size_order", "young_customer", "high_return_history",
                     "size_sensitive_category"]
    brack_inc = increments["bracketing"]
    brack_cpt = {}
    for multi, young, history, cat in iter_product(YN, repeat=4):
        if multi == "Yes":
            base = brack_inc["base_with_multi_size"]
            if young == "Yes":   base += brack_inc["young_customer_with"]
            if history == "Yes": base += brack_inc["high_return_history_with"]
            if cat == "Yes":     base += brack_inc["size_sensitive_category_with"]
        else:
            base = brack_inc["base_without_multi_size"]
            if young == "Yes":   base += brack_inc["young_customer_without"]
            if history == "Yes": base += brack_inc["high_return_history_without"]
            if cat == "Yes":     base += brack_inc["size_sensitive_category_without"]
        base = min(base, 0.95)
        brack_cpt[(multi, young, history, cat)] = {"Yes": round(base, 3),
                                                    "No": round(1 - base, 3)}
    bn.add_node("bracketing", brack_parents, brack_cpt, YN)

    # QUALITY/DAMAGE — P(quality_or_damage | slow_delivery, multiple_items, premium)
    # [1] Damage 10%; [2] 13% faulty goods; [24] ~20% damaged shipping
    qual_parents = ["slow_delivery", "multiple_items_in_order", "premium_price"]
    qual_inc = increments["quality_or_damage"]
    qual_cpt = {}
    for slow, multi, premium in iter_product(YN, repeat=3):
        base = qual_inc["base"]
        if slow == "Yes":    base += qual_inc["slow_delivery"]
        if multi == "Yes":   base += qual_inc["multiple_items_in_order"]
        if premium == "Yes": base += qual_inc["premium_price"]
        base = min(base, 0.95)
        qual_cpt[(slow, multi, premium)] = {"Yes": round(base, 3),
                                             "No": round(1 - base, 3)}
    bn.add_node("quality_or_damage", qual_parents, qual_cpt, YN)

    # -----------------------------------------------------------------
    # LAYER 3: Outcome — P(returned | root causes)
    # -----------------------------------------------------------------
    # Noisy-OR model calibrated so marginal P(returned=Yes) ≈ 0.28-0.33
    # [10] McKinsey: 25% overall; [2] Radial: 30% online clothing

    ret_parents = ROOT_CAUSE_NODES
    ret_cpt = {}

    LEAK = outcome["leak"]
    CAUSE_STRENGTH = outcome["strengths"]

    for combo in iter_product(YN, repeat=5):
        cause_dict = dict(zip(ROOT_CAUSE_NODES, combo))
        # Noisy-OR: P(returned=No) = (1-leak) × ∏(1-strength_i) for active causes
        p_no_return = (1 - LEAK)
        for cause, val in cause_dict.items():
            if val == "Yes":
                p_no_return *= (1 - CAUSE_STRENGTH[cause])
        p_return = 1 - p_no_return
        p_return = max(0.01, min(0.99, p_return))
        ret_cpt[combo] = {"Yes": round(p_return, 4),
                          "No": round(1 - p_return, 4)}

    bn.add_node("returned", ret_parents, ret_cpt, YN)

    return bn


# =============================================================================
# CUSTOMER SEGMENT MODEL (Extended Network)
# =============================================================================
# Addresses the observable independence assumption by adding a latent
# "customer segment" node that induces correlations between signals.
#
# Without this: P(young | social=Yes) = P(young) = 55% (independent)
# With this:    P(young | social=Yes) = 72% (correlated via shared segment)
#
# The segment node is binary-encoded as (seg_a, seg_b) to preserve
# the joint table approach. 4 segments → 2 bits → 20 total nodes.
# Joint table: 2^20 = 1,048,576 entries (~8MB). Build: ~14s, query: ~16ms.

CUSTOMER_SEGMENTS = OrderedDict({
    "gen_z_impulse": {
        "label": "Gen Z Impulse Shopper",
        "prior": 0.25,
        "encoding": (0, 0),  # seg_a=No, seg_b=No
        "description": "Young, social-media-driven, mobile-first, discount-motivated",
    },
    "premium_seeker": {
        "label": "Premium Seeker",
        "prior": 0.15,
        "encoding": (0, 1),  # seg_a=No, seg_b=Yes
        "description": "Quality-focused, desktop-leaning, rarely discounted",
    },
    "serial_bracketer": {
        "label": "Serial Bracketer",
        "prior": 0.10,
        "encoding": (1, 0),  # seg_a=Yes, seg_b=No
        "description": "Returning customer who regularly orders multiple sizes",
    },
    "mainstream": {
        "label": "Mainstream Shopper",
        "prior": 0.50,
        "encoding": (1, 1),  # seg_a=Yes, seg_b=Yes
        "description": "Average shopping behavior, moderate across all signals",
    },
})

# P(observable | segment) — how each segment shifts observable priors
# Calibrated so that marginal P(obs) ≈ DEFAULT_PARAMS priors when mixed.
SEGMENT_PROFILES = {
    "gen_z_impulse": {
        "young_customer": 0.92, "social_media_referral": 0.55, "mobile_purchase": 0.88,
        "purchased_on_discount": 0.60, "is_first_purchase": 0.45, "premium_price": 0.08,
        "multi_size_order": 0.12, "high_return_history": 0.20, "size_sensitive_category": 0.55,
        "viewed_size_guide": 0.15, "slow_delivery": 0.20, "multiple_items_in_order": 0.40,
    },
    "premium_seeker": {
        "young_customer": 0.35, "social_media_referral": 0.15, "mobile_purchase": 0.45,
        "purchased_on_discount": 0.15, "is_first_purchase": 0.30, "premium_price": 0.75,
        "multi_size_order": 0.05, "high_return_history": 0.10, "size_sensitive_category": 0.60,
        "viewed_size_guide": 0.40, "slow_delivery": 0.20, "multiple_items_in_order": 0.25,
    },
    "serial_bracketer": {
        "young_customer": 0.70, "social_media_referral": 0.30, "mobile_purchase": 0.75,
        "purchased_on_discount": 0.45, "is_first_purchase": 0.15, "premium_price": 0.18,
        "multi_size_order": 0.55, "high_return_history": 0.65, "size_sensitive_category": 0.70,
        "viewed_size_guide": 0.30, "slow_delivery": 0.20, "multiple_items_in_order": 0.50,
    },
    "mainstream": {
        "young_customer": 0.42, "social_media_referral": 0.13, "mobile_purchase": 0.57,
        "purchased_on_discount": 0.37, "is_first_purchase": 0.37, "premium_price": 0.13,
        "multi_size_order": 0.02, "high_return_history": 0.08, "size_sensitive_category": 0.48,
        "viewed_size_guide": 0.25, "slow_delivery": 0.20, "multiple_items_in_order": 0.32,
    },
}


def build_segment_network(custom_params=None):
    """
    Construct the extended network with a latent customer segment node.

    Adds a 4-value customer segment (binary-encoded as seg_a, seg_b) that
    induces correlations between observable signals. The root cause CPTs
    and Noisy-OR outcome model are identical to the standard network.

    Network topology (20 nodes, 36 edges):

    SEGMENT (latent)      OBSERVABLE SIGNALS          ROOT CAUSES        OUTCOME
    ════════════════      ═══════════════════         ═══════════        ═══════
    seg_a ──┐
            ├──→ size_sensitive_category ─┐
    seg_b ──┤    is_first_purchase ───────┼─→ size_mismatch ─────┐
            ├──→ viewed_size_guide ───────┤                       │
            ├──→ mobile_purchase ─────────┘                       │
            ├──→ premium_price ───────────┐                       │
            │    is_first_purchase ───────┼─→ expectation_gap ────┤
            │    mobile_purchase ─────────┘                       │
            ├──→ purchased_on_discount ───┐                       ├─→ returned
            ├──→ social_media_referral ───┼─→ impulse_regret ─────┤
            │    mobile_purchase ─────────┤                       │
            ├──→ young_customer ──────────┘                       │
            ├──→ multi_size_order ────────┐                       │
            │    young_customer ──────────┼─→ bracketing ─────────┤
            ├──→ high_return_history ─────┘                       │
            ├──→ slow_delivery ───────────┐                       │
            ├──→ multiple_items_in_order ─┼─→ quality_or_damage ──┘
            │    premium_price ───────────┘
            └──→ ... (all 12 observables)

    Returns
    -------
    BayesNet
        Extended network with customer segment. 20 nodes, 2^20 = 1,048,576 states.
    """
    bn = BayesNet()
    YN = ["Yes", "No"]

    params = get_params(custom_params)
    increments = params["increments"]
    outcome = params["outcome"]

    # --- SEGMENT ENCODING ---
    # Binary-encode 4 segments as (seg_a, seg_b):
    # (No, No)=gen_z, (No, Yes)=premium, (Yes, No)=serial, (Yes, Yes)=mainstream
    seg_priors = {s: CUSTOMER_SEGMENTS[s]["prior"] for s in CUSTOMER_SEGMENTS}
    p_seg_a_yes = seg_priors["serial_bracketer"] + seg_priors["mainstream"]
    p_seg_b_given_a_no = seg_priors["premium_seeker"] / (seg_priors["gen_z_impulse"] + seg_priors["premium_seeker"])
    p_seg_b_given_a_yes = seg_priors["mainstream"] / (seg_priors["serial_bracketer"] + seg_priors["mainstream"])

    bn.add_node("seg_a", [], {
        (): {"Yes": round(p_seg_a_yes, 4), "No": round(1 - p_seg_a_yes, 4)}
    }, YN)
    bn.add_node("seg_b", ["seg_a"], {
        ("Yes",): {"Yes": round(p_seg_b_given_a_yes, 4), "No": round(1 - p_seg_b_given_a_yes, 4)},
        ("No",): {"Yes": round(p_seg_b_given_a_no, 4), "No": round(1 - p_seg_b_given_a_no, 4)},
    }, YN)

    # --- OBSERVABLES (conditioned on segment) ---
    for obs in OBSERVABLE_NODES:
        obs_cpt = {}
        for sa, sb in iter_product(YN, repeat=2):
            if sa == "No" and sb == "No":
                seg = "gen_z_impulse"
            elif sa == "No" and sb == "Yes":
                seg = "premium_seeker"
            elif sa == "Yes" and sb == "No":
                seg = "serial_bracketer"
            else:
                seg = "mainstream"
            p_yes = SEGMENT_PROFILES[seg][obs]
            obs_cpt[(sa, sb)] = {"Yes": round(p_yes, 4), "No": round(1 - p_yes, 4)}
        bn.add_node(obs, ["seg_a", "seg_b"], obs_cpt, YN)

    # --- ROOT CAUSES (identical to standard network) ---
    # SIZE MISMATCH
    size_parents = ["size_sensitive_category", "is_first_purchase",
                    "viewed_size_guide", "mobile_purchase", "premium_price"]
    size_inc = increments["size_mismatch"]
    size_cpt = {}
    for cat, first, guide, mobile, premium in iter_product(YN, repeat=5):
        base = size_inc["base"]
        if cat == "Yes":     base += size_inc["size_sensitive_category"]
        if first == "Yes":   base += size_inc["is_first_purchase"]
        if guide == "Yes":   base += size_inc["viewed_size_guide"]
        if mobile == "Yes":  base += size_inc["mobile_purchase"]
        if premium == "Yes": base += size_inc["premium_price"]
        base = min(base, 0.95)
        size_cpt[(cat, first, guide, mobile, premium)] = {"Yes": round(base, 3), "No": round(1 - base, 3)}
    bn.add_node("size_mismatch", size_parents, size_cpt, YN)

    # EXPECTATION GAP
    exp_parents = ["premium_price", "is_first_purchase", "mobile_purchase",
                   "social_media_referral"]
    exp_inc = increments["expectation_gap"]
    exp_cpt = {}
    for price, first, mobile, social in iter_product(YN, repeat=4):
        base = exp_inc["base"]
        if price == "Yes":  base += exp_inc["premium_price"]
        if first == "Yes":  base += exp_inc["is_first_purchase"]
        if mobile == "Yes": base += exp_inc["mobile_purchase"]
        if social == "Yes": base += exp_inc["social_media_referral"]
        base = min(base, 0.95)
        exp_cpt[(price, first, mobile, social)] = {"Yes": round(base, 3), "No": round(1 - base, 3)}
    bn.add_node("expectation_gap", exp_parents, exp_cpt, YN)

    # IMPULSE REGRET
    imp_parents = ["purchased_on_discount", "social_media_referral",
                   "mobile_purchase", "young_customer"]
    imp_inc = increments["impulse_regret"]
    imp_cpt = {}
    for disc, social, mobile, young in iter_product(YN, repeat=4):
        base = imp_inc["base"]
        if disc == "Yes":    base += imp_inc["purchased_on_discount"]
        if social == "Yes":  base += imp_inc["social_media_referral"]
        if mobile == "Yes":  base += imp_inc["mobile_purchase"]
        if young == "Yes":   base += imp_inc["young_customer"]
        base = min(base, 0.95)
        imp_cpt[(disc, social, mobile, young)] = {"Yes": round(base, 3), "No": round(1 - base, 3)}
    bn.add_node("impulse_regret", imp_parents, imp_cpt, YN)

    # BRACKETING
    brack_parents = ["multi_size_order", "young_customer", "high_return_history",
                     "size_sensitive_category"]
    brack_inc = increments["bracketing"]
    brack_cpt = {}
    for multi, young, history, cat in iter_product(YN, repeat=4):
        if multi == "Yes":
            base = brack_inc["base_with_multi_size"]
            if young == "Yes":   base += brack_inc["young_customer_with"]
            if history == "Yes": base += brack_inc["high_return_history_with"]
            if cat == "Yes":     base += brack_inc["size_sensitive_category_with"]
        else:
            base = brack_inc["base_without_multi_size"]
            if young == "Yes":   base += brack_inc["young_customer_without"]
            if history == "Yes": base += brack_inc["high_return_history_without"]
            if cat == "Yes":     base += brack_inc["size_sensitive_category_without"]
        base = min(base, 0.95)
        brack_cpt[(multi, young, history, cat)] = {"Yes": round(base, 3), "No": round(1 - base, 3)}
    bn.add_node("bracketing", brack_parents, brack_cpt, YN)

    # QUALITY/DAMAGE
    qual_parents = ["slow_delivery", "multiple_items_in_order", "premium_price"]
    qual_inc = increments["quality_or_damage"]
    qual_cpt = {}
    for slow, multi, premium in iter_product(YN, repeat=3):
        base = qual_inc["base"]
        if slow == "Yes":    base += qual_inc["slow_delivery"]
        if multi == "Yes":   base += qual_inc["multiple_items_in_order"]
        if premium == "Yes": base += qual_inc["premium_price"]
        base = min(base, 0.95)
        qual_cpt[(slow, multi, premium)] = {"Yes": round(base, 3), "No": round(1 - base, 3)}
    bn.add_node("quality_or_damage", qual_parents, qual_cpt, YN)

    # OUTCOME — Noisy-OR (identical to standard network)
    ret_parents = ROOT_CAUSE_NODES
    ret_cpt = {}
    LEAK = outcome["leak"]
    CAUSE_STRENGTH = outcome["strengths"]
    for combo in iter_product(YN, repeat=5):
        cause_dict = dict(zip(ROOT_CAUSE_NODES, combo))
        p_no_return = (1 - LEAK)
        for cause, val in cause_dict.items():
            if val == "Yes":
                p_no_return *= (1 - CAUSE_STRENGTH[cause])
        p_return = 1 - p_no_return
        p_return = max(0.01, min(0.99, p_return))
        ret_cpt[combo] = {"Yes": round(p_return, 4), "No": round(1 - p_return, 4)}
    bn.add_node("returned", ret_parents, ret_cpt, YN)

    return bn


def infer_customer_segment(bn, evidence):
    """
    Infer the most probable customer segment from observed signals.

    Only works with the extended (segment) network — requires seg_a, seg_b nodes.

    Parameters
    ----------
    bn : BayesNet
        The segment-extended network (from build_segment_network).
    evidence : dict
        Observed signals, e.g. {"returned": "Yes", "young_customer": "Yes", ...}

    Returns
    -------
    list of dict
        Sorted by posterior (highest first), each containing:
        - name: segment key
        - label: display label
        - posterior: P(segment | evidence)
        - prior: P(segment)
    """
    if "seg_a" not in bn.nodes:
        return []  # Not a segment network

    p_sa = bn.enumeration_ask("seg_a", evidence)
    p_sb_given_sa_yes = bn.enumeration_ask("seg_b", dict(evidence, seg_a="Yes"))
    p_sb_given_sa_no = bn.enumeration_ask("seg_b", dict(evidence, seg_a="No"))

    results = []
    for seg_name, seg_info in CUSTOMER_SEGMENTS.items():
        a_val, b_val = seg_info["encoding"]  # (0,0), (0,1), (1,0), (1,1)
        p_a = p_sa["Yes"] if a_val == 1 else p_sa["No"]
        if a_val == 1:
            p_b = p_sb_given_sa_yes["Yes"] if b_val == 1 else p_sb_given_sa_yes["No"]
        else:
            p_b = p_sb_given_sa_no["Yes"] if b_val == 1 else p_sb_given_sa_no["No"]
        posterior = p_a * p_b

        results.append({
            "name": seg_name,
            "label": seg_info["label"],
            "description": seg_info["description"],
            "posterior": round(posterior, 4),
            "prior": seg_info["prior"],
        })

    results.sort(key=lambda x: x["posterior"], reverse=True)
    return results

def diagnose_return(bn, evidence):
    """
    Run backward inference to diagnose root cause of a return.

    Given observed order signals + returned=Yes, compute posterior
    probability of each root cause and diagnostic lift.

    Parameters
    ----------
    bn : BayesNet
        The return diagnosis Bayesian Network.
    evidence : dict
        Observed signals, e.g. {"size_sensitive_category": "Yes", ...}
        Should include "returned": "Yes" for diagnosis mode.

    Returns
    -------
    list of dict
        Sorted by diagnostic lift (highest first), each containing:
        - name: root cause variable name
        - label: display label
        - prior: P(cause=Yes) baseline
        - posterior: P(cause=Yes | evidence)
        - lift: posterior / prior
        - citation: source reference
    """
    results = []

    for cause in ROOT_CAUSE_NODES:
        # Prior: P(cause=Yes) with no evidence
        prior_dist = bn.enumeration_ask(cause, {})
        prior = prior_dist.get("Yes", 0.0)

        # Posterior: P(cause=Yes | evidence)
        posterior_dist = bn.enumeration_ask(cause, evidence)
        posterior = posterior_dist.get("Yes", 0.0)

        # Diagnostic lift
        lift = posterior / prior if prior > 0 else 0.0

        results.append({
            "name": cause,
            "label": NODE_META[cause]["label"],
            "prior": round(prior, 4),
            "posterior": round(posterior, 4),
            "lift": round(lift, 2),
            "citation": NODE_META[cause]["citation"],
            "base_rate": NODE_META[cause].get("base_rate", ""),
        })

    # Sort by diagnostic lift (most diagnostic first)
    results.sort(key=lambda x: x["lift"], reverse=True)
    return results


def _diagnose_top_cause_fast(bn, evidence):
    """
    Lightweight diagnosis that only returns the top cause name + posterior.
    Uses recursive enumeration (no joint table) — faster for single-use
    networks in parameter_robustness where each variation rebuilds.
    """
    best_name = None
    best_lift = -1
    best_posterior = 0
    for cause in ROOT_CAUSE_NODES:
        prior_dist = bn.recursive_enumeration_ask(cause, {})
        prior = prior_dist.get("Yes", 0.0)
        posterior_dist = bn.recursive_enumeration_ask(cause, evidence)
        posterior = posterior_dist.get("Yes", 0.0)
        lift = posterior / prior if prior > 0 else 0.0
        if lift > best_lift:
            best_lift = lift
            best_name = cause
            best_posterior = posterior
    return best_name, best_posterior, best_lift


def compute_marginal_return_rate(bn):
    """Compute P(returned=Yes) with no evidence — the baseline return rate."""
    dist = bn.enumeration_ask("returned", {})
    return dist.get("Yes", 0.0)


def what_if_analysis(bn, evidence, intervention_node, intervention_value):
    """
    Simulate an intervention: What happens to return probability if we
    change one observable signal?

    This demonstrates the BN's causal reasoning capability.
    (Note: This is observational conditioning, not do-calculus.
    For true causal intervention, graph surgery would be needed.)
    """
    # Current return probability
    current = bn.enumeration_ask("returned", evidence)
    p_return_current = current.get("Yes", 0.0)

    # Intervention: set the node to the new value
    new_evidence = dict(evidence)
    new_evidence[intervention_node] = intervention_value
    new_dist = bn.enumeration_ask("returned", new_evidence)
    p_return_new = new_dist.get("Yes", 0.0)

    return {
        "current_return_prob": round(p_return_current, 4),
        "new_return_prob": round(p_return_new, 4),
        "change": round(p_return_new - p_return_current, 4),
        "change_pct": round((p_return_new - p_return_current) / max(p_return_current, 0.001) * 100, 1),
    }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def signal_sensitivity(bn, base_evidence=None):
    """
    One-at-a-time sensitivity: for each observable signal, compute
    P(returned=Yes | signal=Yes) vs P(returned=Yes | signal=No).

    The swing shows how much each individual signal shifts return probability.
    Also tracks which root cause ranks #1 under each condition.
    """
    results = []
    for node in OBSERVABLE_NODES:
        if base_evidence and node in base_evidence:
            continue

        # P(returned=Yes | signal=Yes)
        ev_yes = {node: "Yes"}
        if base_evidence:
            ev_yes.update(base_evidence)
        p_return_yes = bn.enumeration_ask("returned", ev_yes).get("Yes", 0)
        diag_yes = diagnose_return(bn, dict(ev_yes, returned="Yes"))

        # P(returned=Yes | signal=No)
        ev_no = {node: "No"}
        if base_evidence:
            ev_no.update(base_evidence)
        p_return_no = bn.enumeration_ask("returned", ev_no).get("Yes", 0)
        diag_no = diagnose_return(bn, dict(ev_no, returned="Yes"))

        swing = p_return_yes - p_return_no

        results.append({
            "node": node,
            "label": NODE_META[node]["label"],
            "p_return_yes": round(p_return_yes, 4),
            "p_return_no": round(p_return_no, 4),
            "swing": round(swing, 4),
            "abs_swing": round(abs(swing), 4),
            "top_cause_yes": diag_yes[0]["label"],
            "top_cause_no": diag_no[0]["label"],
            "top_cause_flip": diag_yes[0]["name"] != diag_no[0]["name"],
        })

    results.sort(key=lambda x: x["abs_swing"], reverse=True)
    return results


def parameter_robustness(bn, evidence, perturbation=0.20, custom_params=None):
    """
    Test how robust the top diagnosis is to ±perturbation on each CPT increment.

    Uses _diagnose_top_cause_fast (recursive enumeration, no joint table)
    to avoid the 2.5s joint table build per variation.
    """
    import copy

    base_params = get_params(custom_params)
    baseline = diagnose_return(bn, evidence)
    baseline_top = baseline[0]["name"]
    baseline_posterior = baseline[0]["posterior"]

    flips = []
    max_swing = 0.0
    params_tested = 0

    # Test each increment parameter
    for cause, inc_dict in base_params["increments"].items():
        for key, original_val in inc_dict.items():
            for direction in [1 + perturbation, 1 - perturbation]:
                params_tested += 1
                test_params = copy.deepcopy(base_params)
                new_val = min(max(original_val * direction, 0.0), 0.95)
                test_params["increments"][cause][key] = round(new_val, 4)

                test_bn = build_return_diagnosis_network(test_params)
                test_top, test_posterior, _ = _diagnose_top_cause_fast(test_bn, evidence)

                swing = abs(test_posterior - baseline_posterior)
                max_swing = max(max_swing, swing)

                if test_top != baseline_top:
                    flips.append({
                        "cause": cause,
                        "param": key,
                        "original": original_val,
                        "perturbed": round(new_val, 4),
                        "direction": f"+{int(perturbation*100)}%" if direction > 1 else f"-{int(perturbation*100)}%",
                        "old_top": NODE_META[baseline_top]["label"],
                        "new_top": NODE_META[test_top]["label"],
                        "new_posterior": round(test_posterior, 4),
                    })

    # Test Noisy-OR strengths
    for cause, strength in base_params["outcome"]["strengths"].items():
        for direction in [1 + perturbation, 1 - perturbation]:
            params_tested += 1
            test_params = copy.deepcopy(base_params)
            new_val = min(max(strength * direction, 0.0), 0.99)
            test_params["outcome"]["strengths"][cause] = round(new_val, 4)

            test_bn = build_return_diagnosis_network(test_params)
            test_top, test_posterior, _ = _diagnose_top_cause_fast(test_bn, evidence)

            swing = abs(test_posterior - baseline_posterior)
            max_swing = max(max_swing, swing)

            if test_top != baseline_top:
                flips.append({
                    "cause": cause,
                    "param": f"noisy_or_strength",
                    "original": strength,
                    "perturbed": round(new_val, 4),
                    "direction": f"+{int(perturbation*100)}%" if direction > 1 else f"-{int(perturbation*100)}%",
                    "old_top": NODE_META[baseline_top]["label"],
                    "new_top": NODE_META[test_top]["label"],
                    "new_posterior": round(test_posterior, 4),
                })

    return {
        "baseline_top": NODE_META[baseline_top]["label"],
        "baseline_posterior": round(baseline_posterior, 4),
        "baseline_lift": round(baseline[0]["lift"], 2),
        "total_params_tested": params_tested,
        "flips": flips,
        "stable": len(flips) == 0,
        "max_posterior_swing": round(max_swing, 4),
    }


def posterior_entropy(results):
    """Shannon entropy over root cause posteriors. Lower = more certain diagnosis.
    Returns normalized entropy in [0, 1]: 0 = one cause dominates, 1 = uniform."""
    posteriors = [r["posterior"] for r in results]
    total = sum(posteriors)
    if total == 0:
        return 1.0
    probs = [p / total for p in posteriors if p > 0]
    raw = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(results)) if len(results) > 1 else 1.0
    return raw / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# QUICK VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("Building Fashion Return Root Cause Diagnosis Network...")
    bn = build_return_diagnosis_network()

    print(f"\nNetwork: {len(bn.variables)} nodes")
    print(f"  Observable: {len(OBSERVABLE_NODES)}")
    print(f"  Root Causes: {len(ROOT_CAUSE_NODES)}")
    print(f"  Outcome: 1")

    print(f"\nMarginal return rate P(returned=Yes): "
          f"{compute_marginal_return_rate(bn):.1%}")

    # Test scenario: Dress, first-time buyer, mobile, no size guide
    test_evidence = {
        "returned": "Yes",
        "size_sensitive_category": "Yes",
        "is_first_purchase": "Yes",
        "mobile_purchase": "Yes",
        "viewed_size_guide": "No",
    }
    print(f"\nTest diagnosis (dress, first-time, mobile, no size guide):")
    results = diagnose_return(bn, test_evidence)
    for r in results:
        print(f"  {r['label']:40s} prior={r['prior']:.3f}  "
              f"posterior={r['posterior']:.3f}  lift={r['lift']:.2f}x")

    # Test: Bracketing scenario
    brack_evidence = {
        "returned": "Yes",
        "multi_size_order": "Yes",
        "young_customer": "Yes",
    }
    print(f"\nBracketing test (multi-size order, young customer):")
    results = diagnose_return(bn, brack_evidence)
    for r in results:
        print(f"  {r['label']:40s} prior={r['prior']:.3f}  "
              f"posterior={r['posterior']:.3f}  lift={r['lift']:.2f}x")
