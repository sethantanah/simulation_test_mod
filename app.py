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
    run_simulation as kw_run_simulation,
)
from core.tests.moods_median import (
    moods_median_modified,
    moods_median_original,
    run_simulation as moods_run_simulation,
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


def _emit_progress(progress_queue, label, value, message, kind="progress"):
    if progress_queue is not None:
        progress_queue.put({"label": label, "value": value, "message": message, "kind": kind})


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
    _emit_progress(progress_queue, "kruskal", 0.0, "Initializing Kruskal-Wallis simulation")

    if n_list is None:
        n_list = [20, 100, 500, 1000, 10000]
    if deltas is None:
        deltas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if effect_sizes is None:
        effect_sizes = [0.0, 0.5, 0.75, 1.0]
    if correlations is None:
        correlations = ["independent", "moderate", "strong"]

    _emit_progress(
        progress_queue,
        "kruskal",
        0.1,
        f"Running {n_simulations} sims × {n_monte_carlo} MC reps",
    )

    def _forward_log(message, progress=None):
        if progress is not None:
            _emit_progress(progress_queue, "kruskal", float(progress), message, kind="progress")
        else:
            _emit_progress(progress_queue, "kruskal", 0.0, message, kind="log")

    try:
        df = kw_run_simulation(
            n_simulations=n_simulations,
            n_monte_carlo_reps=n_monte_carlo,
            n_list=n_list,
            deltas=deltas,
            effect_sizes=effect_sizes,
            component_correlations=correlations,
            alpha=alpha,
            base_seed=42,
            methods=["robust", "interval"],
            progress_callback=_forward_log,
        )

        _emit_progress(progress_queue, "kruskal", 0.9, "Formatting results")

        output_path = OUTPUT_DIR / "kruskal_wallis_results.csv"
        df.to_csv(output_path, index=False)

        _emit_progress(progress_queue, "kruskal", 1.0, f"Saved {output_path.name}")
        return {"df": df, "path": output_path}
    except Exception as e:
        _emit_progress(progress_queue, "kruskal", 1.0, f"Error: {str(e)}")
        return {"df": pd.DataFrame(), "path": None}


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
    _emit_progress(progress_queue, "moods", 0.0, "Initializing Mood's Median simulation")

    if n_list is None:
        n_list = [20, 100, 500, 1000, 10000]
    if deltas is None:
        deltas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if effect_sizes is None:
        effect_sizes = [0.0, 0.5, 0.75, 1.0]
    if correlations is None:
        correlations = ["independent", "moderate", "strong"]

    _emit_progress(
        progress_queue,
        "moods",
        0.1,
        f"Running {n_simulations} sims × {n_monte_carlo} MC reps",
    )

    def _forward_log(message, progress=None):
        if progress is not None:
            _emit_progress(progress_queue, "moods", float(progress), message, kind="progress")
        else:
            _emit_progress(progress_queue, "moods", 0.0, message, kind="log")

    try:
        df = moods_run_simulation(
            n_simulations=n_simulations,
            n_monte_carlo_reps=n_monte_carlo,
            n_list=n_list,
            deltas=deltas,
            effect_sizes=effect_sizes,
            component_correlations=correlations,
            alpha=alpha,
            base_seed=42,
            progress_callback=_forward_log,
        )

        _emit_progress(progress_queue, "moods", 0.9, "Formatting results")

        output_path = OUTPUT_DIR / "moods_median_correct_framework.csv"
        df.to_csv(output_path, index=False)

        _emit_progress(progress_queue, "moods", 1.0, f"Saved {output_path.name}")
        return {"df": df, "path": output_path}
    except Exception as e:
        _emit_progress(progress_queue, "moods", 1.0, f"Error: {str(e)}")
        return {"df": pd.DataFrame(), "path": None}


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
    log_lines = {"kruskal": [], "moods": []}

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Kruskal-Wallis")
            status_k = st.empty()
            bar_k = st.progress(0.0)
            log_box_k = st.empty()
        with col2:
            st.subheader("Mood's Median")
            status_m = st.empty()
            bar_m = st.progress(0.0)
            log_box_m = st.empty()

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
                    if update.get("kind") == "log":
                        log_lines[label].append(update["message"])
                        if len(log_lines[label]) > 80:
                            log_lines[label] = log_lines[label][-80:]
                    else:
                        progress_state[label] = float(update["value"])
                        progress_messages[label] = update["message"]

                if progress_state["kruskal"] >= 0.0:
                    bar_k.progress(progress_state["kruskal"])
                    status_k.write(f"{progress_messages['kruskal']} ({progress_state['kruskal'] * 100:.0f}%)")
                    log_box_k.code("\n".join(log_lines["kruskal"][-20:]) if log_lines["kruskal"] else "No logs yet")
                if progress_state["moods"] >= 0.0:
                    bar_m.progress(progress_state["moods"])
                    status_m.write(f"{progress_messages['moods']} ({progress_state['moods'] * 100:.0f}%)")
                    log_box_m.code("\n".join(log_lines["moods"][-20:]) if log_lines["moods"] else "No logs yet")
                time.sleep(0.2)

        while not progress_queue.empty():
            update = progress_queue.get_nowait()
            label = update["label"]
            if update.get("kind") == "log":
                log_lines[label].append(update["message"])
                if len(log_lines[label]) > 80:
                    log_lines[label] = log_lines[label][-80:]
            else:
                progress_state[label] = float(update["value"])
                progress_messages[label] = update["message"]

        bar_k.progress(progress_state["kruskal"])
        status_k.write(f"{progress_messages['kruskal']} ({progress_state['kruskal'] * 100:.0f}%)")
        log_box_k.code("\n".join(log_lines["kruskal"][-20:]) if log_lines["kruskal"] else "No logs yet")
        bar_m.progress(progress_state["moods"])
        status_m.write(f"{progress_messages['moods']} ({progress_state['moods'] * 100:.0f}%)")
        log_box_m.code("\n".join(log_lines["moods"][-20:]) if log_lines["moods"] else "No logs yet")

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
