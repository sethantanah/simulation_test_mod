import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from core.tests.kruskal_wallis import (
    build_neutrosophic_group,
    kruskal_wallis_neutrosophic_interval,
    kruskal_wallis_robust,
    run_simulation,
)
from core.tests.moods_median import (
    moods_median_modified,
    moods_median_original,
    simulate_neutrosophic_data,
)

st.set_page_config(
    page_title="Parallel Neutrosophic Tests",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _emit_progress(progress_queue, label, value, message):
    if progress_queue is not None:
        progress_queue.put({"label": label, "value": value, "message": message})


def _parse_number_list(value, dtype=float, default=None):
    if isinstance(value, list):
        return [dtype(v) for v in value]
    if not isinstance(value, str):
        return default if default is not None else []
    items = [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
    try:
        return [dtype(x) for x in items]
    except ValueError:
        return default if default is not None else []


def run_kruskal_wallis_analysis(
    progress_queue=None,
    n_simulations=1000,
    n_monte_carlo=5,
    n_list=None,
    deltas=None,
    effect_sizes=None,
    correlations=None,
    alpha=0.05,
):
    _emit_progress(progress_queue, "kruskal", 0.0, "Preparing Kruskal-Wallis inputs")

    if n_list is None:
        n_list = [20, 100, 500, 1000, 10000]
    if deltas is None:
        deltas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if effect_sizes is None:
        effect_sizes = [0.0, 0.5, 0.75, 1.0]
    if correlations is None:
        correlations = ["independent", "moderate", "strong"]

    np.random.seed(42)

    np.random.seed(42)

    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]
    g3 = [11, 12, 13, 14, 15]

    crisp = [
        build_neutrosophic_group(g1, falsity_mode="zero"),
        build_neutrosophic_group(g2, falsity_mode="zero"),
        build_neutrosophic_group(g3, falsity_mode="zero"),
    ]

    uncertain = [
        build_neutrosophic_group(g1, 0.1, falsity_mode="complement"),
        build_neutrosophic_group(g2, 0.2, falsity_mode="complement"),
        build_neutrosophic_group(g3, 0.3, falsity_mode="complement"),
    ]

    _emit_progress(
        progress_queue,
        "kruskal",
        0.2,
        f"Using config: {n_simulations} sims, {n_monte_carlo} MC reps, α={alpha}",
    )
    classical_H, classical_p = stats.kruskal(g1, g2, g3)
    robust_crisp = kruskal_wallis_robust(crisp)
    interval_crisp = kruskal_wallis_neutrosophic_interval(crisp)

    _emit_progress(progress_queue, "kruskal", 0.6, "Running uncertain-data checks")
    robust_uncertain = kruskal_wallis_robust(uncertain)
    interval_uncertain = kruskal_wallis_neutrosophic_interval(uncertain)

    rows = [
        {
            "scenario": "crisp",
            "method": "classical",
            "H_statistic": float(classical_H),
            "p_value": float(classical_p),
            "decision": "Reject H0" if classical_p < 0.05 else "Fail to Reject H0",
        },
        {
            "scenario": "crisp",
            "method": "robust",
            "H_statistic": float(robust_crisp.get("H_statistic", np.nan)),
            "p_value": float(robust_crisp.get("p_value", np.nan)),
            "decision": robust_crisp.get("decision", ""),
        },
        {
            "scenario": "crisp",
            "method": "interval",
            "H_statistic": float(interval_crisp.get("H_classical", np.nan)),
            "p_value": float(interval_crisp.get("p_value", np.nan)),
            "decision": interval_crisp.get("decision", ""),
        },
        {
            "scenario": "uncertain",
            "method": "robust",
            "H_statistic": float(robust_uncertain.get("H_statistic", np.nan)),
            "p_value": float(robust_uncertain.get("p_value", np.nan)),
            "decision": robust_uncertain.get("decision", ""),
        },
        {
            "scenario": "uncertain",
            "method": "interval",
            "H_statistic": float(interval_uncertain.get("H_classical", np.nan)),
            "p_value": float(interval_uncertain.get("p_value", np.nan)),
            "decision": interval_uncertain.get("decision", ""),
        },
        {
            "scenario": "simulation_config",
            "method": "settings",
            "H_statistic": np.nan,
            "p_value": np.nan,
            "decision": "configured",
            "n_simulations": n_simulations,
            "n_monte_carlo_reps": n_monte_carlo,
            "n_list": str(n_list),
            "deltas": str(deltas),
            "effect_sizes": str(effect_sizes),
            "component_correlations": str(correlations),
            "alpha": alpha,
        },
    ]

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "kruskal_wallis_results.csv"
    df.to_csv(output_path, index=False)

    _emit_progress(progress_queue, "kruskal", 1.0, f"Saved {output_path.name}")
    return {"df": df, "path": output_path}


def run_moods_median_analysis(
    progress_queue=None,
    n_simulations=100,
    n_monte_carlo=5,
    n_list=None,
    deltas=None,
    effect_sizes=None,
    correlations=None,
    alpha=0.05,
):
    _emit_progress(progress_queue, "moods", 0.0, "Initializing Mood's Median simulation config")

    if n_list is None:
        n_list = [20, 100, 500, 1000, 10000]
    if deltas is None:
        deltas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if effect_sizes is None:
        effect_sizes = [0.0, 0.5, 0.75, 1.0]
    if correlations is None:
        correlations = ["independent", "moderate", "strong"]

    results = []
    total_conditions = len(n_list) * len(deltas) * len(effect_sizes) * len(correlations)
    total_steps = total_conditions * n_monte_carlo
    completed_steps = 0

    for n in n_list:
        for delta_val in deltas:
            for es in effect_sizes:
                for corr in correlations:
                    mc_orig = {"rejections": [], "indeterminates": [], "successes": []}
                    mc_mod = {"rejections": [], "indeterminates": [], "successes": []}

                    for mc_rep in range(n_monte_carlo):
                        rep_seed = 42 + mc_rep * 10000 + completed_steps * 100

                        for sim in range(n_simulations):
                            sim_seed = rep_seed + sim
                            groups = simulate_neutrosophic_data(
                                n_groups=3,
                                n_per_group=n,
                                effect_size=es,
                                indeterminacy_level=delta_val,
                                distribution="normal",
                                component_correlation=corr,
                                seed=sim_seed,
                            )

                            orig = moods_median_original(groups, alpha=alpha)
                            mod = moods_median_modified(groups, alpha=alpha, n_permutations=1000, permutation_seed=sim_seed)

                            if orig is not None:
                                mc_orig["rejections"].append(float(orig["p_value"] < alpha))
                                mc_orig["indeterminates"].append(float(orig.get("is_indeterminate", False)))
                                mc_orig["successes"].append(1)

                            if mod is not None:
                                p_value = mod.get("p_3xk", mod.get("p_lo", mod.get("p_hi", np.nan)))
                                mc_mod["rejections"].append(float(p_value < alpha) if np.isfinite(p_value) else float(mod.get("reject_H0", False)))
                                mc_mod["indeterminates"].append(float(mod.get("is_indeterminate", False)))
                                mc_mod["successes"].append(1)

                        completed_steps += 1
                        progress_value = completed_steps / max(total_steps, 1)
                        _emit_progress(
                            progress_queue,
                            "moods",
                            progress_value,
                            f"Running {n}-sample condition {completed_steps}/{total_steps}",
                        )

                    results.append(
                        {
                            "variant": "original",
                            "component_correlation": corr,
                            "delta": delta_val,
                            "n": n,
                            "effect_size": es,
                            "rejection_rate_mean": float(np.mean(mc_orig["rejections"])) if mc_orig["rejections"] else np.nan,
                            "rejection_rate_std": float(np.std(mc_orig["rejections"])) if mc_orig["rejections"] else np.nan,
                            "indeterminate_rate_mean": float(np.mean(mc_orig["indeterminates"])) if mc_orig["indeterminates"] else np.nan,
                            "indeterminate_rate_std": float(np.std(mc_orig["indeterminates"])) if mc_orig["indeterminates"] else np.nan,
                            "avg_successful_sims": float(np.mean(mc_orig["successes"])) if mc_orig["successes"] else np.nan,
                        }
                    )
                    results.append(
                        {
                            "variant": "modified",
                            "component_correlation": corr,
                            "delta": delta_val,
                            "n": n,
                            "effect_size": es,
                            "rejection_rate_mean": float(np.mean(mc_mod["rejections"])) if mc_mod["rejections"] else np.nan,
                            "rejection_rate_std": float(np.std(mc_mod["rejections"])) if mc_mod["rejections"] else np.nan,
                            "indeterminate_rate_mean": float(np.mean(mc_mod["indeterminates"])) if mc_mod["indeterminates"] else np.nan,
                            "indeterminate_rate_std": float(np.std(mc_mod["indeterminates"])) if mc_mod["indeterminates"] else np.nan,
                            "avg_successful_sims": float(np.mean(mc_mod["successes"])) if mc_mod["successes"] else np.nan,
                        }
                    )

    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "moods_median_correct_framework.csv"
    df.to_csv(output_path, index=False)

    _emit_progress(progress_queue, "moods", 1.0, f"Saved {output_path.name}")
    return {"df": df, "path": output_path}


st.title("Parallel neutrosophic statistical analysis")
st.caption("This page uses the defaults from the __main__ blocks of both test modules and runs them side by side.")

if "kw_settings" not in st.session_state:
    st.session_state.kw_settings = {
        "n_simulations": 1000,
        "n_monte_carlo": 5,
        "n_list": "20,100,500,1000,10000",
        "deltas": "0.0,0.1,0.25,0.5,0.75,1.0",
        "effect_sizes": "0.0,0.5,0.75,1.0",
        "correlations": ["independent", "moderate", "strong"],
        "alpha": 0.05,
    }

if "moods_settings" not in st.session_state:
    st.session_state.moods_settings = {
        "n_simulations": 100,
        "n_monte_carlo": 5,
        "n_list": "20,100,500,1000,10000",
        "deltas": "0.0,0.1,0.25,0.5,0.75,1.0",
        "effect_sizes": "0.0,0.5,0.75,1.0",
        "correlations": ["independent", "moderate", "strong"],
        "alpha": 0.05,
    }

with st.sidebar:
    st.header("Run settings")
    st.info("Adjust simulation parameters for both Kruskal-Wallis and Mood's Median.")

    with st.expander("Kruskal-Wallis settings", expanded=True):
        st.session_state.kw_settings["n_simulations"] = st.number_input(
            "Kruskal N_SIMULATIONS",
            min_value=1,
            value=st.session_state.kw_settings["n_simulations"],
            step=1,
        )
        st.session_state.kw_settings["n_monte_carlo"] = st.number_input(
            "Kruskal N_MONTE_CARLO",
            min_value=1,
            value=st.session_state.kw_settings["n_monte_carlo"],
            step=1,
        )
        st.session_state.kw_settings["n_list"] = st.text_input(
            "Kruskal N_LIST",
            value=st.session_state.kw_settings["n_list"],
        )
        st.session_state.kw_settings["deltas"] = st.text_input(
            "Kruskal DELTAS",
            value=st.session_state.kw_settings["deltas"],
        )
        st.session_state.kw_settings["effect_sizes"] = st.text_input(
            "Kruskal EFFECT_SIZES",
            value=st.session_state.kw_settings["effect_sizes"],
        )
        st.session_state.kw_settings["correlations"] = st.multiselect(
            "Kruskal CORRELATIONS",
            options=["independent", "moderate", "strong"],
            default=st.session_state.kw_settings["correlations"],
        )
        st.session_state.kw_settings["alpha"] = st.slider(
            "Kruskal ALPHA",
            min_value=0.001,
            max_value=0.1,
            value=float(st.session_state.kw_settings["alpha"]),
            step=0.001,
        )

    with st.expander("Mood's Median settings", expanded=True):
        st.session_state.moods_settings["n_simulations"] = st.number_input(
            "Mood's N_SIMULATIONS",
            min_value=1,
            value=st.session_state.moods_settings["n_simulations"],
            step=1,
        )
        st.session_state.moods_settings["n_monte_carlo"] = st.number_input(
            "Mood's N_MONTE_CARLO",
            min_value=1,
            value=st.session_state.moods_settings["n_monte_carlo"],
            step=1,
        )
        st.session_state.moods_settings["n_list"] = st.text_input(
            "Mood's N_LIST",
            value=st.session_state.moods_settings["n_list"],
        )
        st.session_state.moods_settings["deltas"] = st.text_input(
            "Mood's DELTAS",
            value=st.session_state.moods_settings["deltas"],
        )
        st.session_state.moods_settings["effect_sizes"] = st.text_input(
            "Mood's EFFECT_SIZES",
            value=st.session_state.moods_settings["effect_sizes"],
        )
        st.session_state.moods_settings["correlations"] = st.multiselect(
            "Mood's CORRELATIONS",
            options=["independent", "moderate", "strong"],
            default=st.session_state.moods_settings["correlations"],
        )
        st.session_state.moods_settings["alpha"] = st.slider(
            "Mood's ALPHA",
            min_value=0.001,
            max_value=0.1,
            value=float(st.session_state.moods_settings["alpha"]),
            step=0.001,
        )

    run_button = st.button("Run both analyses in parallel", use_container_width=True)


if run_button:
    st.session_state["running"] = True

if st.session_state.get("running", False):
    progress_queue = queue.Queue()
    progress_state = {"kruskal": 0.0, "moods": 0.0}
    progress_messages = {"kruskal": "Waiting to start", "moods": "Waiting to start"}

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Kruskal-Wallis")
            status_k = st.empty()
            bar_k = st.progress(0.0)
        with col2:
            st.subheader("Mood's Median")
            status_m = st.empty()
            bar_m = st.progress(0.0)

        kw_n_list = _parse_number_list(
            st.session_state.kw_settings["n_list"], dtype=int,
            default=[20, 100, 500, 1000, 10000],
        )
        kw_deltas = _parse_number_list(
            st.session_state.kw_settings["deltas"], dtype=float,
            default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        )
        kw_effect_sizes = _parse_number_list(
            st.session_state.kw_settings["effect_sizes"], dtype=float,
            default=[0.0, 0.5, 0.75, 1.0],
        )
        kw_correlations = st.session_state.kw_settings["correlations"] or [
            "independent", "moderate", "strong"
        ]

        moods_n_list = _parse_number_list(
            st.session_state.moods_settings["n_list"], dtype=int,
            default=[20, 100, 500, 1000, 10000],
        )
        moods_deltas = _parse_number_list(
            st.session_state.moods_settings["deltas"], dtype=float,
            default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        )
        moods_effect_sizes = _parse_number_list(
            st.session_state.moods_settings["effect_sizes"], dtype=float,
            default=[0.0, 0.5, 0.75, 1.0],
        )
        moods_correlations = st.session_state.moods_settings["correlations"] or [
            "independent", "moderate", "strong"
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_kw = executor.submit(
                run_kruskal_wallis_analysis,
                progress_queue,
                st.session_state.kw_settings["n_simulations"],
                st.session_state.kw_settings["n_monte_carlo"],
                kw_n_list,
                kw_deltas,
                kw_effect_sizes,
                kw_correlations,
                st.session_state.kw_settings["alpha"],
            )
            future_mm = executor.submit(
                run_moods_median_analysis,
                progress_queue,
                st.session_state.moods_settings["n_simulations"],
                st.session_state.moods_settings["n_monte_carlo"],
                moods_n_list,
                moods_deltas,
                moods_effect_sizes,
                moods_correlations,
                st.session_state.moods_settings["alpha"],
            )

            while not (future_kw.done() and future_mm.done()):
                while not progress_queue.empty():
                    update = progress_queue.get_nowait()
                    label = update["label"]
                    progress_state[label] = float(update["value"])
                    progress_messages[label] = update["message"]

                if progress_state["kruskal"] >= 0.0:
                    bar_k.progress(progress_state["kruskal"])
                    status_k.write(f"{progress_messages['kruskal']} ({progress_state['kruskal'] * 100:.0f}%)")
                if progress_state["moods"] >= 0.0:
                    bar_m.progress(progress_state["moods"])
                    status_m.write(f"{progress_messages['moods']} ({progress_state['moods'] * 100:.0f}%)")
                time.sleep(0.2)

        while not progress_queue.empty():
            update = progress_queue.get_nowait()
            label = update["label"]
            progress_state[label] = float(update["value"])
            progress_messages[label] = update["message"]

        bar_k.progress(progress_state["kruskal"])
        status_k.write(f"{progress_messages['kruskal']} ({progress_state['kruskal'] * 100:.0f}%)")
        bar_m.progress(progress_state["moods"])
        status_m.write(f"{progress_messages['moods']} ({progress_state['moods'] * 100:.0f}%)")

    kruskal_result = future_kw.result()
    moods_result = future_mm.result()

    st.session_state["kruskal_result"] = kruskal_result
    st.session_state["moods_result"] = moods_result
    st.session_state["running"] = False


if "kruskal_result" in st.session_state:
    kruskal_result = st.session_state["kruskal_result"]
    moods_result = st.session_state["moods_result"]

    st.success("Both analyses finished successfully.")

    tab1, tab2 = st.tabs(["Kruskal-Wallis", "Mood's Median"])

    with tab1:
        st.dataframe(kruskal_result["df"], use_container_width=True)
        st.download_button(
            label="Download Kruskal-Wallis CSV",
            data=kruskal_result["path"].read_bytes(),
            file_name=kruskal_result["path"].name,
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"Saved to: {kruskal_result['path']}")

    with tab2:
        st.dataframe(moods_result["df"], use_container_width=True)
        st.download_button(
            label="Download Mood's Median CSV",
            data=moods_result["path"].read_bytes(),
            file_name=moods_result["path"].name,
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"Saved to: {moods_result['path']}")

else:
    st.info("Press the button above to run both analyses and download the generated CSV files.")
