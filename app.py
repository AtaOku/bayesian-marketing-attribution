"""
Fashion E-Commerce Return Root Cause Diagnosis
================================================
Streamlit Application — Academic Project + Portfolio Demo

Personal project applying Bayesian Network concepts from
        Fundamentals of Artificial Intelligence (IN2406), TUM

Author: Ata Okuzcuoglu
        MSc Management & Technology (Marketing + CS)
"""

import streamlit as st
import time
import io
import csv
from src.bayesian_return_diagnosis import (
    build_return_diagnosis_network,
    diagnose_return,
    compute_marginal_return_rate,
    what_if_analysis,
    signal_sensitivity,
    NODE_META,
    OBSERVABLE_NODES,
    ROOT_CAUSE_NODES,
    OBSERVABLE_GROUPS,
    DEFAULT_PARAMS,
    get_params,
    params_to_flat_csv_rows,
    flat_csv_rows_to_params,
    validate_params,
    CONTINUOUS_INPUTS,
    compute_weight,
    continuous_to_evidence_and_weights,
    apply_intensity_weights,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Fashion Return Root Cause Diagnosis — Bayesian Network",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .academic-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .academic-header h1 { color: white !important; font-size: 1.7rem !important; margin-bottom: 0.3rem !important; }
    .academic-header .subtitle { color: #a8b2d1; font-size: 0.95rem; margin-bottom: 0.8rem; }
    .academic-header .course-info {
        color: #8892b0; font-size: 0.82rem;
        border-top: 1px solid #233554; padding-top: 0.6rem; margin-top: 0.6rem;
    }
    .cause-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1.2rem; margin-bottom: 0.8rem; transition: all 0.2s ease;
    }
    .cause-card:hover { border-color: #667eea; box-shadow: 0 2px 12px rgba(102,126,234,0.12); }
    .cause-card.top-cause { border-left: 4px solid #e53e3e; background: #fff5f5; }
    .cause-name { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.3rem; }
    .cause-citation {
        font-size: 0.75rem; color: #718096; font-style: italic;
        margin-top: 0.5rem; border-top: 1px solid #edf2f7; padding-top: 0.4rem;
    }
    .lift-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
    .lift-high { background: #fed7d7; color: #c53030; }
    .lift-med  { background: #fefcbf; color: #975a16; }
    .lift-low  { background: #c6f6d5; color: #276749; }
    .ref-item { font-size: 0.8rem; color: #4a5568; margin-bottom: 0.4rem; padding-left: 1.5rem; text-indent: -1.5rem; }
    .method-box {
        background: #fffff0; border: 1px solid #ecc94b; border-radius: 8px;
        padding: 1rem 1.2rem; font-size: 0.88rem; margin: 0.5rem 0;
    }
    .whatif-card {
        background: linear-gradient(135deg, #ebf8ff 0%, #e9d8fd 100%);
        border: 1px solid #bee3f8; border-radius: 10px; padding: 1.2rem;
    }
    .sidebar-group-title {
        font-size: 0.85rem; font-weight: 600; color: #4a5568;
        margin-top: 1rem; margin-bottom: 0.3rem;
        padding-bottom: 0.2rem; border-bottom: 1px solid #e2e8f0;
    }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# BUILD NETWORK (supports custom parameters via session state)
# =============================================================================
@st.cache_resource
def get_network(params_hash=None, custom_params=None):
    """Build network and pre-compute joint table. params_hash is for cache invalidation."""
    net = build_return_diagnosis_network(custom_params)
    net._ensure_joint()  # Eagerly build joint table so subsequent queries are instant
    return net


def _params_hash(params):
    """Create a hashable key from params dict for cache invalidation."""
    if params is None:
        return "default"
    import json
    return hash(json.dumps(params, sort_keys=True))


# Initialize custom params in session state
if "custom_params" not in st.session_state:
    st.session_state.custom_params = None

custom_params = st.session_state.custom_params
bn = get_network(params_hash=_params_hash(custom_params), custom_params=custom_params)


# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="academic-header" style="padding: 0.8rem 1.5rem;">
    <h3 style="margin:0 0 0.3rem 0;">Fashion E-Commerce Return Root Cause Diagnosis</h3>
    <div class="subtitle" style="font-size:0.88rem;">
        A Bayesian Network that infers <em>why</em> a customer returned a fashion item — backward inference from order signals to hidden root causes.
    </div>
</div>
""", unsafe_allow_html=True)

# Custom parameters indicator
if st.session_state.get("custom_params") is not None:
    custom_mr = compute_marginal_return_rate(bn)
    default_bn_ref = get_network(params_hash="default", custom_params=None)
    default_mr_ref = compute_marginal_return_rate(default_bn_ref)
    st.markdown(f"""
    <div style="background: #f0fff4; border: 1px solid #68d391; border-radius: 8px;
                padding: 0.6rem 1.2rem; margin-bottom: 1rem; font-size: 0.85rem;">
        <strong>Custom parameters active</strong> — Calibrated return rate: <strong>{custom_mr:.1%}</strong>
        (default: {default_mr_ref:.1%}) · Go to <strong>Calibration</strong> tab to adjust or reset.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================
tab_diagnose, tab_whatif, tab_sensitivity, tab_calibrate, tab_method, tab_refs = st.tabs([
    "🔬 Diagnosis", "🧪 What-If Simulation", "📊 Sensitivity Analysis",
    "⚙️ Calibration", "📐 Methodology & Network", "📚 References"
])


# =============================================================================
# TAB 1: DIAGNOSIS
# =============================================================================
with tab_diagnose:

    with st.expander("**What is this?** A Bayesian Network that diagnoses *why* customers return orders", expanded=False):
        st.markdown("""
        <div style="font-size: 0.92rem; color: #4a5568; line-height: 1.7;">
            <strong>Your analytics dashboard shows <em>which</em> orders get returned.</strong>
            This tool tells you <strong>why</strong> — inferring hidden root causes from signals already in your
            order system (product category, customer type, device, discount, delivery time).
            <br><br>
            Pick a scenario on the left or set custom signals in the sidebar. The Bayesian Network below
            runs backward inference to rank the most probable cause.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.image("assets/bn_topology.png", use_container_width=True)
        st.caption(
            "**Reading this diagram:** Grey boxes = observable signals. "
            "Colored boxes = 5 hidden root causes. Edge labels = CPT risk increments. "
            "**q values** = Noisy-OR strengths (how likely each cause triggers a return). "
            "Full details in the **Methodology & Network** tab."
        )

    st.markdown("#### Select a scenario to diagnose why the order was returned")

    # --- Scenario Selection (main area) ---
    SCENARIO_INFO = {
        "The Size-Blind First-Timer": {
            "subtitle": "First-time buyer orders a dress on mobile without checking the size guide",
            "signals": {"size_sensitive_category": "Yes", "is_first_purchase": "Yes",
                        "mobile_purchase": "Yes", "viewed_size_guide": "No",
                        "multi_size_order": "No"},
            "expect": "Size/Fit Mismatch — high-risk category, no size guide, new customer",
        },
        "The Instagram Impulse Shopper": {
            "subtitle": "Gen Z grabs a flash sale from a social media ad on their phone",
            "signals": {"purchased_on_discount": "Yes", "social_media_referral": "Yes",
                        "mobile_purchase": "Yes", "young_customer": "Yes",
                        "is_first_purchase": "Yes", "multi_size_order": "No"},
            "expect": "Impulse/Buyer's Regret — discount-driven, unplanned purchase",
        },
        "The Try-At-Home Bracketer": {
            "subtitle": "Gen Z discovers brand via Instagram, orders 2 sizes of the same item",
            "signals": {"is_first_purchase": "Yes", "young_customer": "Yes",
                        "social_media_referral": "Yes", "mobile_purchase": "Yes",
                        "multi_size_order": "Yes"},
            "expect": "Bracketing — ordered two sizes to try at home",
        },
        "The Disappointed Premium Buyer": {
            "subtitle": "New customer buys an expensive item on mobile — high expectations unmet",
            "signals": {"premium_price": "Yes", "is_first_purchase": "Yes",
                        "mobile_purchase": "Yes", "multi_size_order": "No"},
            "expect": "Expectation Gap — premium price sets high quality bar",
        },
        "The Frustrated Late Receiver": {
            "subtitle": "Multi-item premium order arrives late — frustration and quality concerns",
            "signals": {"slow_delivery": "Yes", "premium_price": "Yes",
                        "multiple_items_in_order": "Yes", "multi_size_order": "No"},
            "expect": "Quality/Fulfillment — slow shipping + premium expectations",
        },
        "The Serial Returner": {
            "subtitle": "Known high-return customer orders multiple sizes on discount",
            "signals": {"high_return_history": "Yes", "young_customer": "Yes",
                        "multi_size_order": "Yes", "purchased_on_discount": "Yes"},
            "expect": "Bracketing — habitual bracket-and-return behavior",
        },
    }

    scenario_names = list(SCENARIO_INFO.keys())

    # Build PRESETS dict from SCENARIO_INFO for backward compat
    PRESETS = {name: info["signals"] for name, info in SCENARIO_INFO.items()}

    # --- Horizontal button selector ---
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = scenario_names[0]

    btn_cols = st.columns(len(scenario_names))
    for i, (name, col) in enumerate(zip(scenario_names, btn_cols)):
        with col:
            is_active = st.session_state.selected_scenario == name
            # Short display name (remove "The ")
            short = name.replace("The ", "")
            if st.button(
                short, key=f"sc_{i}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.selected_scenario = name
                st.rerun()

    preset = st.session_state.selected_scenario

    # Scenario detail card
    if preset in SCENARIO_INFO:
        info = SCENARIO_INFO[preset]
        active_signals = [k for k, v in info["signals"].items() if v == "Yes"]
        signal_tags = "".join(
            f'<span style="background:#ebf4ff; color:#2b6cb0; padding:2px 8px; '
            f'border-radius:12px; font-size:0.78rem; margin:2px;">'
            f'{NODE_META[s]["label"]}</span>' for s in active_signals
        )
        st.markdown(f"""
        <div style="background:#f7fafc; border:1px solid #e2e8f0; border-radius:8px;
                    padding:0.5rem 1rem; margin:0.3rem 0 0.8rem 0; display:flex; align-items:center; flex-wrap:wrap; gap:6px;">
            <span style="color:#4a5568; font-size:0.85rem;">{info['subtitle']}</span>
            <span style="color:#cbd5e0;">|</span>
            {signal_tags}
        </div>
        """, unsafe_allow_html=True)

    # --- Sidebar: signals only ---
    with st.sidebar:
        st.markdown("### Order Signals")
        is_custom = st.toggle(
            "Custom signals",
            value=False,
            key="custom_mode",
            help="Override the preset and set each signal manually",
        )
        if is_custom:
            st.caption("Set observable signals for a returned order. "
                       "Unknown signals are marginalized automatically.")

            # Continuous input toggle
            continuous_mode = st.toggle(
                "Continuous Mode",
                value=False,
                key="continuous_mode",
                help="Enter exact values (discount %, age, price €, delivery days) "
                     "instead of Yes/No. The model adjusts CPT weights based on intensity.",
            )
        else:
            continuous_mode = False

        # --- Build evidence ---
        evidence = {"returned": "Yes"}
        active_evidence = {}
        continuous_values = {}

        if not is_custom and preset in PRESETS:
            active_evidence = PRESETS[preset]
            evidence.update(active_evidence)

        # Track which nodes are handled by continuous input
        continuous_nodes = set(CONTINUOUS_INPUTS.keys()) if continuous_mode else set()

        for group_name, nodes in OBSERVABLE_GROUPS.items():
            st.markdown(f'<div class="sidebar-group-title">{group_name}</div>',
                        unsafe_allow_html=True)
            for node in nodes:
                meta = NODE_META[node]

                if is_custom:
                    if node in continuous_nodes:
                        # --- Continuous input: number slider ---
                        ci = CONTINUOUS_INPUTS[node]
                        val = st.slider(
                            f"{ci['label']} ({ci['unit']})",
                            min_value=ci["min_val"],
                            max_value=ci["max_val"],
                            value=ci["default_val"],
                            step=ci["step"],
                            key=f"ci_{node}",
                            help=ci["help"],
                        )
                        continuous_values[node] = val

                        # Show computed weight
                        weight = compute_weight(val, ci["ref_points"])
                        if ci["direction"] == "above":
                            is_yes = val > ci["threshold"]
                        else:
                            is_yes = val < ci["threshold"]
                        status = "Yes" if is_yes else "No"
                        st.caption(f"→ {status} · weight: {weight:.2f}×")

                        if is_yes:
                            evidence[node] = "Yes"
                            active_evidence[node] = "Yes"
                        elif val == ci["min_val"] if ci["direction"] == "above" else val == ci["max_val"]:
                            # Explicit "no" — user set to minimum/maximum
                            evidence[node] = "No"
                            active_evidence[node] = "No"
                        # else: leave as Unknown (marginalized)
                    else:
                        # --- Binary input: selectbox ---
                        val = st.selectbox(
                            meta["label"],
                            ["N/A", "Yes", "No"],
                            key=f"ev_{node}",
                            help=f"{meta['description']}\n\n{meta['citation']}",
                        )
                        if val != "N/A":
                            evidence[node] = val
                            active_evidence[node] = val
                else:
                    preset_val = active_evidence.get(node, "N/A")
                    st.selectbox(
                        meta["label"],
                        [preset_val],
                        key=f"ev_{node}_{preset[:8]}",
                        disabled=True,
                        help=f"{meta['description']}\n\n{meta['citation']}",
                    )

        st.markdown("---")
        n_continuous = len([n for n in continuous_values if n in active_evidence])
        n_binary = len(active_evidence) - n_continuous
        if continuous_mode and n_continuous > 0:
            st.caption(f"**{len(active_evidence)}** signals set "
                       f"({n_continuous} continuous, {n_binary} binary) · "
                       f"**{len(OBSERVABLE_NODES) - len(active_evidence)}** marginalized")
        else:
            st.caption(f"**{len(active_evidence)}** signals set · "
                       f"**{len(OBSERVABLE_NODES) - len(active_evidence)}** marginalized")

    # --- Apply intensity weights if continuous mode active ---
    diagnosis_bn = bn  # Default: use the top-level network (with calibration params)

    if continuous_values:
        _, weights = continuous_to_evidence_and_weights(continuous_values)
        if any(abs(w - 1.0) > 0.01 for w in weights.values()):
            # At least one weight differs from default → rebuild with modulated params
            base_params = get_params(st.session_state.custom_params)
            modulated_params = apply_intensity_weights(base_params, weights)
            diagnosis_bn = build_return_diagnosis_network(modulated_params)

    # --- Run Diagnosis ---
    t0 = time.time()
    results = diagnose_return(diagnosis_bn, evidence)
    inference_time = (time.time() - t0) * 1000

    marginal_return = compute_marginal_return_rate(diagnosis_bn)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Baseline Return Rate", f"{marginal_return:.1%}",
                   help="Average return rate across all orders — P(returned=Yes) with no signals observed")
    with col_m2:
        st.metric("Evidence Signals", f"{len(active_evidence)} / {len(OBSERVABLE_NODES)}",
                   help="Signals set vs marginalized")
    with col_m3:
        st.metric("Inference Time", f"{inference_time:.0f} ms",
                   help="Exact enumeration over 2¹⁸ = 262,144 joint states")

    # Show continuous weight detail if active
    if continuous_values:
        _, active_weights = continuous_to_evidence_and_weights(continuous_values)
        non_default = {k: v for k, v in active_weights.items() if abs(v - 1.0) > 0.01}
        if non_default:
            weight_parts = []
            for node, w in non_default.items():
                ci = CONTINUOUS_INPUTS[node]
                val = continuous_values[node]
                weight_parts.append(f"{ci['label']}: {val}{ci['unit']} → {w:.2f}×")
            st.caption("**Continuous weights:** " + " · ".join(weight_parts))

    st.markdown("---")
    st.subheader("Root Cause Diagnosis")

    if not active_evidence:
        st.warning("No evidence set. Select a scenario in the sidebar or switch to Custom mode.")

    for i, r in enumerate(results):
        is_top = (i == 0 and r["lift"] > 1.5 and len(active_evidence) > 0)
        if r["lift"] >= 2.0:
            lift_class = "lift-high"
        elif r["lift"] >= 1.3:
            lift_class = "lift-med"
        else:
            lift_class = "lift-low"

        card_class = "cause-card top-cause" if is_top else "cause-card"
        bar_pct = min(r["posterior"] * 100, 100)
        bar_color = "#e53e3e" if is_top else "#667eea"

        st.markdown(f"""
        <div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="cause-name">{"" if is_top else ""}{r['label']}</div>
                <div><span class="lift-badge {lift_class}">{r['lift']:.2f}x lift</span></div>
            </div>
            <div style="display:flex; gap:2rem; margin-top:0.5rem; font-size:0.88rem;">
                <div><span style="color:#718096;">Prior:</span> <strong>{r['prior']:.1%}</strong></div>
                <div><span style="color:#718096;">Posterior:</span> <strong>{r['posterior']:.1%}</strong></div>
            </div>
            <div style="background:#edf2f7; border-radius:6px; height:14px; margin-top:0.6rem; overflow:hidden;">
                <div style="background:{bar_color}; height:100%; width:{bar_pct}%; border-radius:4px;
                            transition: width 0.5s ease;"></div>
            </div>
            <div class="cause-citation">{r['citation']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Actionable insight
    if active_evidence and results[0]["lift"] > 1.5:
        top = results[0]
        st.markdown("---")
        st.subheader("Recommended Intervention")

        interventions = {
            "size_mismatch": (
                "**Improve size guidance for this customer segment.**\n\n"
                "- Add mandatory size guide popup for first-time mobile customers buying fit-sensitive categories\n"
                "- Implement AI size recommendation based on past purchases (for returning customers)\n"
                "- Show fit-specific customer reviews ('True to size' / 'Runs small') prominently\n\n"
                "*[8] MIT Sloan: Form-fitting garments returned more than casual; diagnostics reveal which "
                "product features drive returns. [9] ScienceDirect: Size finder users are slightly more likely "
                "to return — suggesting the tool signals uncertainty rather than resolving it.*"
            ),
            "expectation_gap": (
                "**Bridge the gap between online presentation and reality.**\n\n"
                "- Add 360° product photos and video with natural lighting\n"
                "- Show fabric close-ups and on-model videos with diverse body types\n"
                "- Include honest customer photo reviews\n\n"
                "*[4] AfterShip / J. Business Economics: 45.9% returned because they 'disliked the item,' "
                "21% because it was 'not as described.' [23] Radial: Accurate modeling, images, description, "
                "and sizing are table stakes.*"
            ),
            "impulse_regret": (
                "**Add friction to impulse-driven purchase paths.**\n\n"
                "- Show 'Are you sure?' confirmation for social-media-referred discount purchases\n"
                "- Add 'Save for later' prominently on mobile discount pages\n"
                "- Implement wishlist nudges instead of immediate checkout for flash sales\n\n"
                "*[23] Radial: 57% of women report impulse-buying clothing online. [3] Rocket Returns: "
                "Social media purchases show the highest return rates of all channels.*"
            ),
            "bracketing": (
                "**Reduce the need to bracket by improving pre-purchase confidence.**\n\n"
                "- Implement virtual try-on or AR fitting tools\n"
                "- Show 'Customers who bought size M are typically 5'6\", 130 lbs'\n"
                "- Offer 'Help me choose my size' quiz before checkout for multi-size carts\n\n"
                "*[5] Landmark Global: 51% of Gen Z regularly bracket purchases. "
                "[7] Statista: 48% bracket when sizing is unclear. [14] Naval Research Logistics: "
                "Bracketing can sometimes benefit retailers (keep rate >75%).*"
            ),
            "quality_or_damage": (
                "**Improve fulfillment quality and packaging.**\n\n"
                "- Audit packaging for multi-item orders (higher damage risk)\n"
                "- Add quality inspection checkpoint before shipping premium items\n"
                "- Investigate slow-delivery carriers for handling issues\n\n"
                "*[1] Coresight: Damage accounts for 10% of returns. "
                "[2] Radial: 13% of returns due to faulty/damaged goods. "
                "[24] Opensend: ~20% of returns stem from shipping damage.*"
            ),
        }
        st.markdown(interventions.get(top["name"], "Review order patterns for this segment."))
        st.caption("These are **illustrative examples** based on industry research — not prescriptions. "
                   "Your optimal intervention depends on your platform, customer base, and business model.")# =============================================================================
# TAB 2: WHAT-IF SIMULATION
# =============================================================================
with tab_whatif:
    st.subheader("What-If Simulation: Test Interventions")
    st.caption("Change one observable signal and observe the effect on return probability "
               "and root cause distribution.")

    with st.expander("Methodological note: observational vs causal inference", expanded=False):
        st.markdown("""
        This uses observational conditioning P(returned | X=x), not true causal intervention
        via do-calculus. For causal intervention, graph surgery (removing incoming edges to X)
        would be needed. See Pearl (2009) *Causality* and Russell & Norvig Ch. 14.5.
        """)

    col_wi1, col_wi2 = st.columns(2)

    with col_wi1:
        st.markdown("**Baseline Scenario:**")
        wi_evidence = {"returned": "Yes"}
        for node in OBSERVABLE_NODES:
            wi_val = st.selectbox(
                NODE_META[node]["label"],
                ["N/A", "Yes", "No"],
                key=f"wi_{node}",
                help=NODE_META[node]["description"],
            )
            if wi_val != "N/A":
                wi_evidence[node] = wi_val

    with col_wi2:
        st.markdown("**Intervention:**")
        intervention_node = st.selectbox(
            "Which signal to change?",
            OBSERVABLE_NODES,
            format_func=lambda x: NODE_META[x]["label"],
        )
        intervention_value = st.radio(
            "Set to:", ["Yes", "No"], horizontal=True, key="wi_val"
        )

        if st.button("Run What-If Analysis", type="primary", use_container_width=True):
            result = what_if_analysis(bn, wi_evidence, intervention_node, intervention_value)

            st.markdown(f"""
            <div class="whatif-card">
                <div style="font-size:0.9rem; margin-bottom:0.8rem;">
                    <strong>Intervention:</strong> Set
                    <code>{NODE_META[intervention_node]['label']}</code> = <code>{intervention_value}</code>
                </div>
                <div style="display:flex; gap:1.5rem; font-size:1.1rem;">
                    <div><span style="color:#718096;">Before:</span> <strong>{result['current_return_prob']:.1%}</strong></div>
                    <div style="font-size:1.5rem;">→</div>
                    <div><span style="color:#718096;">After:</span> <strong>{result['new_return_prob']:.1%}</strong></div>
                    <div>
                        <span style="color:#718096;">Δ</span>
                        <strong style="color:{'#38a169' if result['change'] < 0 else '#e53e3e'};">
                            {result['change_pct']:+.1f}%
                        </strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            new_ev = dict(wi_evidence)
            new_ev[intervention_node] = intervention_value
            new_results = diagnose_return(bn, new_ev)

            st.markdown("**Updated Root Cause Distribution:**")
            wi_max_posterior = max(r["posterior"] for r in new_results) if new_results else 1.0
            for r in new_results:
                bar_pct = (r["posterior"] / wi_max_posterior * 100) if wi_max_posterior > 0 else 0
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.8rem; margin:0.3rem 0; font-size:0.88rem;">
                    <div style="width:220px;">{r['label']}</div>
                    <div style="flex:1; background:#edf2f7; border-radius:4px; height:16px; overflow:hidden;">
                        <div style="background:#667eea; height:100%; width:{bar_pct}%;
                                    border-radius:4px; display:flex; align-items:center;
                                    padding-left:6px; color:white; font-size:0.72rem; font-weight:600;">
                            {r['posterior']:.1%}
                        </div>
                    </div>
                    <div style="width:60px; text-align:right; font-weight:600;">{r['lift']:.1f}x</div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: SENSITIVITY ANALYSIS
# =============================================================================
with tab_sensitivity:
    st.subheader("Sensitivity Analysis")
    st.caption("How much does each signal matter? How robust is the diagnosis to parameter uncertainty?")

    st.markdown("""
    <div class="method-box">
    <strong>Why sensitivity analysis?</strong> A diagnosis is only useful if it's robust.
    If the same signal dominates regardless of how you set others, that's a strong signal to act on.
    Each of the 12 observable signals is toggled Yes vs No (one at a time, with returned=Yes)
    to measure its individual impact on return probability.
    </div>
    """, unsafe_allow_html=True)

    mr = compute_marginal_return_rate(bn)
    st.markdown("#### Which signal has the biggest impact on return probability?")
    st.caption(f"Baseline P(returned=Yes) = {mr:.0%}. Each signal is toggled Yes vs No independently "
               "to measure its individual effect on return probability.")

    # Cache key based on params
    _sens_key = f"sens_{hash(str(st.session_state.get('custom_params')))}"
    if _sens_key not in st.session_state:
        with st.spinner("Running sensitivity across all 12 signals..."):
            st.session_state[_sens_key] = signal_sensitivity(bn)
    sens_results = st.session_state[_sens_key]

    # Tornado chart using native Streamlit
    if sens_results:
        import plotly.graph_objects as go

        labels = [r["label"] for r in sens_results]
        p_yes_vals = [r["p_return_yes"] for r in sens_results]
        p_no_vals = [r["p_return_no"] for r in sens_results]
        swings = [r["swing"] for r in sens_results]

        # Horizontal bar chart: Yes vs No
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels,
            x=[r["p_return_yes"] for r in sens_results],
            name="Signal = Yes",
            orientation="h",
            marker_color="#e53e3e",
            text=[f'{v:.1%}' for v in p_yes_vals],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            y=labels,
            x=[r["p_return_no"] for r in sens_results],
            name="Signal = No",
            orientation="h",
            marker_color="#38a169",
            text=[f'{v:.1%}' for v in p_no_vals],
            textposition="auto",
        ))
        fig.add_vline(
            x=mr, line_dash="dash", line_color="#718096", line_width=1.5,
            annotation_text=f"Baseline: {mr:.0%}", annotation_position="top",
        )
        fig.update_layout(
            barmode="group",
            title="Return Probability",
            xaxis_title="Return Probability",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(labels) * 45),
            margin=dict(l=200),
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Swing table
        st.markdown("#### Signal Swing Ranking")
        swing_data = []
        for r in sens_results:
            swing_data.append({
                "Signal": r["label"],
                "P(return|Yes)": f'{r["p_return_yes"]:.1%}',
                "P(return|No)": f'{r["p_return_no"]:.1%}',
                "Swing": f'{r["swing"]:+.1%}',
                "Top Cause": r["top_cause_yes"],
            })

        st.dataframe(swing_data, use_container_width=True, hide_index=True)

        st.caption(f"Baseline P(returned=Yes) = {mr:.1%}. "
                   "Adjust parameters in the Calibration tab — this analysis updates automatically.")


# =============================================================================
# TAB 4: CALIBRATION — "Does this work with my own data?"
# =============================================================================
with tab_calibrate:
    st.subheader("Parameter Calibration")
    st.caption("Adjust parameters to match your own data, or upload a CSV. "
               "All other tabs update live with your custom parameters.")
    st.caption("**Privacy:** Your parameters and uploaded CSVs are processed in-memory only — "
               "nothing is stored, logged, or sent to any server. When you close this tab, the data is gone.")

    st.markdown("""
    <div class="method-box">
    <strong>Why calibrate?</strong> The default parameters are grounded in industry research
    (17 citations) — but your platform likely differs. A fast-fashion brand has ~38% return rates;
    a luxury brand has ~18%. Mobile traffic might be 78% or 45%. This tab lets you plug in
    <em>your</em> numbers and see how the diagnosis changes.
    </div>
    """, unsafe_allow_html=True)

    # --- Active params display ---
    using_custom = st.session_state.custom_params is not None
    if using_custom:
        st.success("**Custom parameters active.** All tabs use your calibrated model.")
        if st.button("Reset to Research Defaults", key="reset_params"):
            st.session_state.custom_params = None
            st.rerun()
    else:
        st.info("**Using research-grounded defaults.** Adjust below to calibrate to your data.")

    # --- Tabs within calibration ---
    cal_mode = st.radio(
        "Calibration mode:",
        ["Quick Mode (Priors + Strengths)", "Advanced (All CPT Increments)", "CSV Import/Export"],
        horizontal=True,
        key="cal_mode",
    )

    current_params = get_params(st.session_state.custom_params)

    # ---------------------------------------------------------------
    # QUICK MODE: Observable priors + Noisy-OR strengths
    # ---------------------------------------------------------------
    if cal_mode == "Quick Mode (Priors + Strengths)":

        st.markdown("#### Observable Signal Priors")
        st.caption("P(signal = Yes) — What fraction of your orders have each characteristic?")

        new_priors = {}
        prior_groups = {
            "Product": ["size_sensitive_category", "premium_price"],
            "Customer": ["is_first_purchase", "young_customer", "high_return_history"],
            "Behavior": ["viewed_size_guide", "mobile_purchase"],
            "Purchase": ["purchased_on_discount", "social_media_referral",
                            "multi_size_order", "multiple_items_in_order"],
            "Fulfillment": ["slow_delivery"],
        }

        for group_name, nodes in prior_groups.items():
            st.markdown(f"**{group_name}**")
            cols = st.columns(len(nodes))
            for i, node in enumerate(nodes):
                default_val = DEFAULT_PARAMS["priors"][node]
                current_val = current_params["priors"][node]
                with cols[i]:
                    new_val = st.slider(
                        NODE_META[node]["label"],
                        min_value=0.01, max_value=0.99,
                        value=current_val,
                        step=0.01,
                        key=f"cal_prior_{node}",
                        help=f"Default: {default_val:.0%} · {NODE_META[node]['citation']}",
                    )
                    new_priors[node] = new_val
                    if abs(new_val - default_val) > 0.005:
                        st.caption(f"Default: {default_val:.0%}")

        st.markdown("---")
        st.markdown("#### Noisy-OR Outcome Parameters")
        st.caption("How likely each active root cause leads to a return.")

        col_leak, col_spacer2 = st.columns([1, 3])
        with col_leak:
            new_leak = st.slider(
                "Leak probability (λ)",
                min_value=0.01, max_value=0.30,
                value=current_params["outcome"]["leak"],
                step=0.01,
                key="cal_leak",
                help=f"Returns with no modeled cause. Default: {DEFAULT_PARAMS['outcome']['leak']:.0%}",
            )

        new_strengths = {}
        str_cols = st.columns(5)
        for i, cause in enumerate(ROOT_CAUSE_NODES):
            default_s = DEFAULT_PARAMS["outcome"]["strengths"][cause]
            current_s = current_params["outcome"]["strengths"][cause]
            with str_cols[i]:
                new_s = st.slider(
                    NODE_META[cause]["label"].split(" ", 1)[1] if " " in NODE_META[cause]["label"] else NODE_META[cause]["label"],
                    min_value=0.10, max_value=0.99,
                    value=current_s,
                    step=0.01,
                    key=f"cal_str_{cause}",
                    help=f"Default: {default_s:.0%}",
                )
                new_strengths[cause] = new_s
                if abs(new_s - default_s) > 0.005:
                    st.caption(f"Default: {default_s:.0%}")

        # --- Apply button ---
        st.markdown("---")
        col_apply, col_preview = st.columns([1, 2])
        with col_apply:
            if st.button("Apply Custom Parameters", type="primary", use_container_width=True,
                          key="apply_quick"):
                new_params = {
                    "priors": new_priors,
                    "outcome": {
                        "leak": new_leak,
                        "strengths": new_strengths,
                    },
                }
                st.session_state.custom_params = new_params
                st.rerun()

        # --- Live preview ---
        with col_preview:
            preview_params = {
                "priors": new_priors,
                "outcome": {"leak": new_leak, "strengths": new_strengths},
            }
            preview_bn = build_return_diagnosis_network(preview_params)
            preview_mr = compute_marginal_return_rate(preview_bn)
            default_bn = build_return_diagnosis_network()
            default_mr = compute_marginal_return_rate(default_bn)
            delta = preview_mr - default_mr

            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Default Marginal Return Rate", f"{default_mr:.1%}")
            pcol2.metric("Your Calibrated Rate", f"{preview_mr:.1%}",
                         delta=f"{delta:+.1%}", delta_color="inverse")

    # ---------------------------------------------------------------
    # ADVANCED MODE: Full CPT increment editing
    # ---------------------------------------------------------------
    elif cal_mode == "Advanced (All CPT Increments)":

        st.markdown("#### CPT Risk Increments")
        st.caption("Each parent's additive contribution to root cause probability. "
                   "Formula: P(cause) = base + Σ(increment × parent=Yes), capped at 0.95.")

        new_increments = {}

        for cause in ROOT_CAUSE_NODES:
            with st.expander(f"**{NODE_META[cause]['label']}**", expanded=False):
                inc = current_params["increments"][cause]
                default_inc = DEFAULT_PARAMS["increments"][cause]
                new_inc = {}

                for param, val in inc.items():
                    default_v = default_inc[param]
                    display_name = param.replace("_", " ").title()
                    new_v = st.slider(
                        display_name,
                        min_value=0.00, max_value=0.95,
                        value=val,
                        step=0.01,
                        key=f"cal_inc_{cause}_{param}",
                        help=f"Default: {default_v}",
                    )
                    new_inc[param] = new_v
                new_increments[cause] = new_inc

        st.markdown("---")
        if st.button("Apply Advanced Parameters", type="primary", key="apply_advanced"):
            # Build full custom params: keep priors and outcome from current,
            # override increments
            new_params = dict(current_params)
            new_params["increments"] = new_increments
            st.session_state.custom_params = new_params
            st.rerun()

    # ---------------------------------------------------------------
    # CSV IMPORT/EXPORT
    # ---------------------------------------------------------------
    else:
        st.markdown("#### Export Current Parameters")
        st.caption("Download a CSV of all parameters. Edit in Excel/Sheets, then re-upload.")

        export_params = get_params(st.session_state.custom_params)
        rows = params_to_flat_csv_rows(export_params)

        # Build CSV string
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["section", "parameter", "value", "default"])

        default_rows = params_to_flat_csv_rows(DEFAULT_PARAMS)
        default_lookup = {(r[0], r[1]): r[2] for r in default_rows}

        for section, param, val in rows:
            default_val = default_lookup.get((section, param), val)
            writer.writerow([section, param, f"{val:.4f}", f"{default_val:.4f}"])

        st.download_button(
            "Download Parameters CSV",
            data=csv_buffer.getvalue(),
            file_name="bn_parameters.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### Import Custom Parameters")
        st.caption("Upload a CSV with columns: `section`, `parameter`, `value`. "
                   "The `default` column is ignored on import (it's for your reference).")

        uploaded = st.file_uploader("Upload parameter CSV", type=["csv"], key="csv_upload")

        if uploaded is not None:
            # File size guard (max 1MB)
            if uploaded.size > 1_000_000:
                st.error("File too large. Maximum size is 1MB.")
            else:
                try:
                    content = uploaded.getvalue().decode("utf-8")
                    reader = csv.DictReader(io.StringIO(content))

                    # Validate required columns
                    if not reader.fieldnames or not all(
                        col in reader.fieldnames for col in ["section", "parameter", "value"]
                    ):
                        st.error("CSV must have columns: `section`, `parameter`, `value`.")
                    else:
                        import_rows = []
                        parse_errors = []
                        for i, row in enumerate(reader):
                            try:
                                import_rows.append((
                                    row["section"].strip(),
                                    row["parameter"].strip(),
                                    float(row["value"])
                                ))
                            except (ValueError, TypeError):
                                parse_errors.append(f"Row {i+2}: could not parse value '{row.get('value', '')}'")

                        if parse_errors:
                            st.warning(f"⚠️ Skipped {len(parse_errors)} unparseable rows.")
                            with st.expander("Show parse warnings"):
                                for pe in parse_errors[:20]:
                                    st.caption(pe)

                        import_params = flat_csv_rows_to_params(import_rows)

                        # Show validation warnings (unknown keys, out-of-range values)
                        csv_warnings = import_params.pop("_warnings", [])
                        if csv_warnings:
                            st.warning(f"⚠️ {len(csv_warnings)} validation warnings.")
                            with st.expander("Show validation warnings"):
                                for w in csv_warnings[:20]:
                                    st.caption(w)

                        # Final validation
                        is_valid, errors = validate_params(import_params)
                        if not is_valid:
                            st.error("Invalid parameter values detected:")
                            for err in errors:
                                st.caption(f"  • {err}")
                        else:
                            # Preview
                            import_bn = build_return_diagnosis_network(import_params)
                            import_mr = compute_marginal_return_rate(import_bn)
                            default_mr = compute_marginal_return_rate(build_return_diagnosis_network())

                            n_applied = sum(
                                len(v) if isinstance(v, dict) else 1
                                for v in [import_params.get("priors", {}),
                                          *import_params.get("increments", {}).values(),
                                          import_params.get("outcome", {}).get("strengths", {})]
                            ) + (1 if "leak" in import_params.get("outcome", {}) else 0)

                            st.success(f"✅ Parsed and validated {n_applied} parameters from CSV.")

                            ic1, ic2 = st.columns(2)
                            ic1.metric("Default Return Rate", f"{default_mr:.1%}")
                            ic2.metric("Imported Return Rate", f"{import_mr:.1%}",
                                       delta=f"{import_mr - default_mr:+.1%}", delta_color="inverse")

                            # Count changes from defaults
                            default_lookup = {(r[0], r[1]): r[2] for r in params_to_flat_csv_rows(DEFAULT_PARAMS)}
                            imported_lookup = {(r[0], r[1]): r[2] for r in params_to_flat_csv_rows(get_params(import_params))}
                            n_changed = sum(1 for k in imported_lookup
                                            if abs(imported_lookup[k] - default_lookup.get(k, imported_lookup[k])) > 0.001)
                            st.caption(f"**{n_changed}** parameters differ from defaults.")

                            if st.button("✅ Apply Imported Parameters", type="primary", key="apply_csv"):
                                st.session_state.custom_params = import_params
                                st.rerun()

                except UnicodeDecodeError:
                    st.error("❌ Could not read file. Please upload a UTF-8 encoded CSV.")
                except Exception:
                    st.error("❌ Unexpected error processing CSV. Please check the format matches the reference below.")

        st.markdown("---")
        st.markdown("#### CSV Format Reference")
        st.code("""section,parameter,value,default
priors,mobile_purchase,0.7800,0.6500
priors,young_customer,0.6000,0.5500
increments,size_mismatch.base,0.0400,0.0400
increments,size_mismatch.size_sensitive_category,0.1500,0.1200
outcome,leak,0.0500,0.0500
outcome,strength.size_mismatch,0.8000,0.8000""", language="csv")


# =============================================================================
# TAB 4: METHODOLOGY & NETWORK
# =============================================================================
with tab_method:
    st.subheader("Methodology")

    st.markdown("#### 1. Problem Statement")
    st.markdown("""
    Fashion e-commerce suffers from return rates of **25-40%**, costing the global industry
    an estimated **$218 billion** annually [2].

    **The critical gap:** When a return happens, the platform knows *that* it happened — but
    **not why**. The return reason is fundamentally unobservable:
    - Customer surveys have **30-40% completion rates** and are often inaccurate [4]
    - Customers misreport reasons (e.g., "didn't like it" when the issue was brand sizing)
    - Return forms offer broad categories ("doesn't fit") that don't distinguish root causes
    - Dashboards show aggregate return rates but cannot diagnose individual orders

    Yet the actual reasons follow consistent patterns [1][2][3]:

    | Root Cause | Share of Returns | Sources |
    |---|---|---|
    | Size / Fit Mismatch | 53–70% | Coresight [1], Radial [2], Rocket Returns [3] |
    | Style / Color / Expectation Gap | 16–23% | Coresight [1], AfterShip [4] |
    | Impulse / Buyer's Regret | 8–15% | Opensend, Radial [23] |
    | Quality / Damage | 10–13% | Coresight [1], Radial [2] |
    | Intentional Bracketing | ~15% (of multi-brand) | Returnalyze [6], Vogue Business |

    This project applies **Bayesian Network backward inference** to infer the most probable
    hidden root cause from observable order-level signals — data already available in any
    e-commerce order management system, **without requiring the customer to report anything**.
    """)

    st.markdown("#### 2. Why Bayesian Networks?")
    st.markdown("""
    The critical distinction is **backward (diagnostic) inference** vs. forward prediction:

    | Capability | BN | Dashboard / BI | Logistic Regression |
    |---|---|---|---|
    | Backward inference (effect → cause) | ✅ Bayes' theorem | ❌ | ❌ |
    | Missing data handling | ✅ Marginalization | ❌ Imputation | ❌ Imputation |
    | Causal structure | ✅ DAG | ❌ Correlation | ❌ Correlation |
    | Interpretability | ✅ CPTs readable | ✅ | ⚠️ Coefficients |
    | What-if reasoning | ✅ | ❌ | ⚠️ Limited |

    **Key insight:** Regression predicts *P(returned | features)* — whether a return will happen.
    A BN computes *P(root_cause | returned=Yes, signals)* — **why** it happened. This is the
    information a merchandising team actually needs to take action.
    """)

    st.markdown("#### 3. Formal Definition")
    st.markdown("A Bayesian Network is a tuple **(X, G, P)** where:")
    st.markdown("""
    - **X** = {X₁, ..., X₁₈}: 18 discrete random variables (all binary)
    - **G** = (V, E): DAG with 18 nodes and 22 directed edges
    - **P** = {P(Xᵢ | Parents(Xᵢ))}: Conditional Probability Tables
    """)
    st.markdown("Inference uses **Enumeration-Ask** (Russell & Norvig, Fig. 14.9):")
    st.latex(r"P(X \mid \mathbf{e}) = \alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y})")
    st.markdown("Diagnostic lift quantifies how much evidence shifts belief about a cause:")
    st.latex(r"\text{lift}(C) = \frac{P(C = \text{Yes} \mid \mathbf{e})}{P(C = \text{Yes})}")
    st.markdown("""**Inference optimization:** The full joint probability table (2¹⁸ = 262,144 entries) is computed
    once and cached. Subsequent queries marginalize over the cached table via NumPy boolean masking —
    reducing each `enumeration_ask` call from O(2ⁿ) recursive expansion to an O(2ⁿ) vectorized sum.
    First call: ~2.5s (joint build). All subsequent queries: <20ms.""")

    st.markdown("#### 4. Network Topology")
    st.markdown("""
    The network has a **three-layer architecture**: observable signals → latent root causes → outcome.
    """)
    st.image("assets/bn_topology.png", use_container_width=True)
    st.markdown("""
    **How to read this diagram:**
    - **Grey boxes** (left): Observable signals with prior probabilities — data already in your order system
    - **Colored boxes** (center): 5 latent root causes with marginal P(cause) and industry return share
    - **Edge labels** (left → center): CPT risk increments — how much each signal raises the cause probability
    - **q values** (center → right): Noisy-OR strengths — how likely each active cause leads to a return
    - **Shared parents** are key: `mobile_purchase` feeds 3 causes, `first_purchase` feeds 2 — these
      dependencies create explaining-away effects that single-variable analysis cannot capture
    """)

    st.markdown("#### 5. CPT Calibration")
    st.markdown("""
    CPTs are calibrated using a **research-grounded additive risk model**:

    1. **Observable priors** from industry benchmarks (e.g., P(mobile) = 0.65 from [3])
    2. **Root cause CPTs** — each parent contributes a calibrated risk increment:
       - Size-sensitive category: +12% to size mismatch [3]
       - First purchase: +8% [10] (McKinsey: 67% higher returns for new customers)
       - Viewed size guide: +3% [9] (counterintuitive — signals uncertainty)
    3. **Outcome** uses **Noisy-OR** (Russell & Norvig, Ch. 14.3):
    """)
    st.latex(r"P(\text{return} = \text{No}) = (1 - \lambda) \prod_{i: C_i = \text{Yes}} (1 - q_i)")

    # Dynamic table — shows current values (may differ from defaults if custom params active)
    active_params = get_params(st.session_state.custom_params)
    active_strengths = active_params["outcome"]["strengths"]
    default_strengths = DEFAULT_PARAMS["outcome"]["strengths"]

    strength_rows = ""
    for cause, label, rationale in [
        ("size_mismatch", "Size Mismatch", "Very likely to return if doesn't fit"),
        ("expectation_gap", "Expectation Gap", "Often return if looks/feels different"),
        ("impulse_regret", "Impulse Regret", "~50% keep despite initial regret"),
        ("bracketing", "Bracketing", "Almost always return extra sizes"),
        ("quality_or_damage", "Quality/Fulfillment", "Very likely to return defective items"),
    ]:
        val = active_strengths[cause]
        default_val = default_strengths[cause]
        marker = f" *(custom — default: {default_val})*" if abs(val - default_val) > 0.005 else ""
        strength_rows += f"    | {label} | {val} | {rationale}{marker} |\n"

    st.markdown(f"""
    | Root Cause | Strength (qᵢ) | Rationale |
    |---|---|---|
{strength_rows}
    **Leak probability** λ = {active_params['outcome']['leak']}
    {f"*(custom — default: {DEFAULT_PARAMS['outcome']['leak']})*" if abs(active_params['outcome']['leak'] - DEFAULT_PARAMS['outcome']['leak']) > 0.005 else "(returns with no modeled cause: gift returns, policy abuse, etc.)"}

    **Calibration result:** Marginal P(returned=Yes) ≈ {compute_marginal_return_rate(bn):.0%}, matching industry range of 25-40% [2][10].

    **→ All 42 parameters (12 priors, 24 CPT increments, 6 outcome parameters) are adjustable
    via the Calibration tab** — including CSV export/import for team workflows. This lets you
    calibrate the model to your own platform's data while keeping the network structure and
    research-grounded defaults as a starting point.
    """)

    st.markdown("#### 6. Network Statistics")
    marginal = compute_marginal_return_rate(bn)
    total_edges = sum(len(bn.nodes[n].parents) for n in bn.variables)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes", "18")
    c2.metric("Total Edges", str(total_edges))
    c3.metric("State Space", "2¹⁸ = 262,144")
    c4.metric("Marginal P(return)", f"{marginal:.1%}")

    if using_custom:
        st.caption("*Statistics reflect your custom parameters. "
                   "Reset in the Calibration tab to see defaults.*")

    st.markdown("#### 7. Sensitivity Analysis")
    st.markdown("""
    The **Sensitivity** tab provides one-at-a-time signal sensitivity analysis:

    Each of the 12 observable signals is toggled Yes vs No independently (with all others
    marginalized), measuring the swing in P(returned=Yes). This reveals which signals have
    the greatest individual impact on return probability.

    **Key finding:** `Multi-Size Order` dominates with a ~50% swing — consistent with bracketing
    being the strongest single-signal predictor. Most other signals produce 1-7% swings,
    confirming that the model requires **multiple signals** to discriminate between causes reliably.
    """)

    st.markdown("#### 8. Structural Constraints")
    st.markdown("""
    **Bracketing requires Multi-Size Order.** Bracketing is defined as deliberately ordering
    multiple sizes to try at home — it is logically impossible without a multi-size order.
    The CPT enforces this: P(bracketing=Yes | multi_size=No, ...) = 0.00 for all parent
    combinations. When `multi_size_order` is unobserved, a small residual probability remains
    via marginalization (8% prior × 77% conditional ≈ 6%), which is correct Bayesian reasoning.
    """)

    st.markdown("#### 9. Limitations & Future Work")
    st.markdown("""
    - **CPTs are expert-calibrated, not learned from data.** The ⚙️ Calibration tab allows
      adjusting priors, CPT increments, and Noisy-OR strengths to match your platform's data.
      With access to real order-level data, parameter learning (MLE or Bayesian estimation,
      Russell & Norvig Ch. 20) could automate this. Structure learning (e.g., NOTEARS, PC
      algorithm) could discover edges.
    - **Observational conditioning ≠ causal intervention.** The What-If tab uses P(Y|X=x),
      not P(Y|do(X=x)). True causal reasoning requires graph surgery (Pearl, 2009).
    - **Binary discretization** simplifies continuous variables. Multi-valued domains
      (e.g., delivery_days: Fast/Normal/Slow) would increase expressiveness at the cost
      of larger CPTs.
    - **No temporal dynamics.** A Dynamic Bayesian Network could model how return behavior
      changes across a customer's lifecycle (see Project 3: HMM-Based Segmentation).
    """)

    st.markdown("#### 8. Input Validation")
    st.markdown("""
    The Calibration tab accepts user-provided parameters via sliders and CSV upload.
    To prevent invalid network states, all inputs are validated:

    - **Range checks:** All probability values must be finite numbers in [0, 1]
    - **Whitelist:** Only the 42 known parameter keys are accepted — unknown keys are rejected
    - **Row limits:** CSV uploads are capped at 200 rows / 1MB to prevent resource exhaustion
    - **Sanitized errors:** Error messages do not expose internal paths or stack traces
    - **Additive overflow cap:** CPT values from the additive risk model are capped at 0.95

    Streamlit's built-in widget constraints (slider min/max, selectbox options) provide
    additional defense-in-depth for interactive inputs.
    """)


# =============================================================================
# TAB 5: REFERENCES
# =============================================================================
with tab_refs:
    st.subheader("📚 References")

    st.markdown("#### Industry Research & Data")
    refs = [
        "[1] **Coresight Research** (2023). 'The True Cost of Apparel Returns: Alarming Return Rates Require Loss-Minimization Solutions.' Size/fit: 53%, color: 16%, damage: 10%. [Survey of US apparel brands/retailers]",
        "[2] **Radial** (2024). 'Tech Takes on E-Commerce's $218 Billion Returns Problem.' Poor fit/sizing: 70% of fashion returns. 13% faulty goods. 16% not matching description.",
        "[3] **Rocket Returns** (2025). 'Ecommerce Return Rates: Complete Industry Analysis + Benchmarks by Category.' Size/fit: 67%, style/color: 23%, quality: 10%. Social media = highest return channel. Mobile elevated vs desktop.",
        "[4] **AfterShip** (2024). 'Returns: Fashion's $218 Billion Problem.' Citing Journal of Business Economics: 45.9% returned because customer disliked item, 21% not as described.",
        "[5] **Landmark Global / IPC Cross-Border E-Commerce Shopper Survey** (2025). 'Wardrobing & Bracketing: How to Manage Serial Returners.' 51% Gen Z regularly bracket; 43% wardrobing. >50% fashion orders returned in DE and UK.",
        "[6] **Returnalyze** (2023). 'Breaking Down Bracketing.' ~15% of multi-brand returns due to bracketing. Keep rate for bracketing >75%. Size-bracketed items have higher return rates than color-bracketed.",
        "[7] **Statista** (2022). 'Top Reasons for Bracketing Online Purchases in the U.S.' 48% bracket when sizing unclear, 36% can't try on in-store, 31% forgot their size, 27% ordered multiple to return.",
        "[8] **MIT Sloan / Hauser et al.** (2024). 'How Better Predictive Models Could Lead to Fewer Clothing Returns.' Form-fitting garments returned more than casual. Horizontal stripes returned less than vertical. Color + images improve return prediction.",
        "[9] **ScienceDirect** (2025). 'Fits Like a Glove? Knowledge and Use of Size Finders and High-End Fashion Retail Returns.' Size finder users 0.65% MORE likely to return. n=496,365 items, 75,707 customers, 113 countries, July 2015–April 2022. Swedish fashion e-commerce platform.",
        "[10] **McKinsey & Company** (2021). 'Returning to Order: Improving Returns Management for Apparel Companies.' 25% return rate apparel e-commerce pre-COVID. Returns management not top-5 priority for 1/3 of retailers. First-time online shoppers: 67% higher return rates.",
    ]
    for r in refs:
        st.markdown(f'<div class="ref-item">{r}</div>', unsafe_allow_html=True)

    st.markdown("#### Academic & Methodological")
    refs2 = [
        "[11] **Russell, S. & Norvig, P.** (2021). *Artificial Intelligence: A Modern Approach*, 4th ed. Ch. 13: Quantifying Uncertainty. Ch. 14: Probabilistic Reasoning. Ch. 20: Learning Probabilistic Models. Pearson.",
        "[12] **MDPI Electronics** (2024). Chen et al. 'Proactive Return Prediction in Online Fashion Retail Using Heterogeneous Graph Neural Networks.' GNN with customer, order, product nodes. Features: order value, items count, day-of-week, month.",
        "[13] **ScienceDirect / Transportation Research Part E** (2024). 'The Billion-Pound Question in Fashion E-Commerce: Investigating the Anatomy of Returns.' UK's 2nd-largest pure-play fashion retailer. £7B annual cost, 750K tonnes CO₂.",
        "[14] **Naval Research Logistics** (2022). Balaram et al. 'Bracketing of Purchases to Manage Size Uncertainty.' First formal model of bracketing. Price as lever for retailer response. Moderate match probability + low hassle cost → encourage bracketing.",
        "[15] **ZigZag Global** (2025). 'How to Combat Serial Returners.' AI-driven customer return scoring. ASOS flags/limits high-return accounts. 66% of fashion retailers now charge for at least one return type.",
        "[16] **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press. do-calculus, graph surgery, causal intervention vs observational conditioning.",
    ]
    for r in refs2:
        st.markdown(f'<div class="ref-item">{r}</div>', unsafe_allow_html=True)

    st.markdown("#### Course Material")
    refs3 = [
        "[17] **TUM IN2406** (2025). Lecture 9: Bayesian Networks. Fundamentals of Artificial Intelligence, TUM.",
        "[18] **TUM IN2406** — Bayesian Networks notebook exercises: exact inference, d-separation, CPT construction, parameter learning.",
    ]
    for r in refs3:
        st.markdown(f'<div class="ref-item">{r}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Default CPTs are calibrated from directional relationships in industry research. "
               "Use the Calibration tab to adjust priors, risk increments, and Noisy-OR strengths "
               "to match your platform's data — or upload a CSV. With access to real order-level data, "
               "parameter learning (MLE/Bayesian estimation) could automate this process "
               "(Russell & Norvig Ch. 20).")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="academic-header" style="padding: 0.8rem 1.5rem; margin-top: 1rem;">
    <div style="font-size: 0.82rem; line-height: 1.8; opacity: 0.9;">
        <strong>Course:</strong> Fundamentals of Artificial Intelligence (IN2406) · TUM<br>
        <strong>Author:</strong> Ata Okuzcuoglu · MSc Management & Technology (Marketing + CS)<br>
        <strong>Context:</strong> Personal project applying course concepts to a real marketing problem<br>
        <strong>Technique:</strong> Bayesian Networks — Exact Enumeration Inference (Russell & Norvig, Ch. 13-14)
    </div>
</div>
""", unsafe_allow_html=True)
