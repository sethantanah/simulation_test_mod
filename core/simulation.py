import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Callable
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray, neutrosophicate
from core.tests.kruskal_wallis import kruskal_wallis_original, kruskal_wallis_modified
from core.tests.mann_whitney import mann_whitney_original, mann_whitney_modified
from core.tests.moods_median import moods_median_original, moods_median_modified

# Maximum finite relative efficiency stored in the DataFrame.
# Infinite values (orig_power == 0, mod_power > 0) are capped here so that
# aggregations (mean, std) remain finite.  The dashboard may display "≥ RE_CAP".
_RE_CAP = 3.0


class MonteCarloSimulation:
    """
    Monte Carlo simulation framework for comparing original and modified
    neutrosophic non-parametric tests.

    Implements the experimental design specified in Objective 3:
    - Type I error rate analysis (under H0)
    - Statistical power analysis (under H1)
    - Decision stability under varying indeterminacy
    - Interval width precision metrics
    - Relative efficiency comparisons
    """

    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        # FIX 1 & 2: self.rng is used ONLY for the indeterminacy index selection
        # inside induce_indeterminacy().  Each simulation iteration spawns its own
        # independent Generator via np.random.default_rng(seed + sim_index), so
        # results are both reproducible and independent across iterations.
        self.rng = np.random.default_rng(seed=random_seed)

    # -------------------------------------------------------------------------
    # Data generation
    # -------------------------------------------------------------------------

    def generate_data(
        self,
        n: int,
        distribution: str,
        effect_size: float,
        k: int = 3,
        rng: Optional[np.random.Generator] = None,
    ) -> List[List[float]]:
        """
        Generate synthetic data for one simulation iteration.

        FIX 1 & 2: accepts an explicit ``rng`` parameter so that each
        simulation iteration passes its own seeded Generator instead of
        sharing self.rng.  When called from outside run_single_condition
        (e.g. quick interactive tests), rng defaults to self.rng.
        """
        if rng is None:
            rng = self.rng

        groups = []
        for i in range(k):
            loc = i * effect_size

            if distribution == "normal":
                data = rng.normal(loc=loc, scale=1.0, size=n)

            elif distribution == "skewed":
                # Lognormal: mean of log-normal = exp(mu + sigma²/2).
                # We shift by loc so groups differ in location.
                data = rng.lognormal(mean=0.0, sigma=0.5, size=n) + loc

            elif distribution == "heavy_tailed":
                data = rng.standard_t(df=3, size=n) + loc

            elif distribution == "uniform":
                # Unit-variance uniform: range = 2*sqrt(3), var = range²/12 = 1
                half = np.sqrt(3)
                data = rng.uniform(low=loc - half, high=loc + half, size=n)

            elif distribution == "bimodal":
                n1 = n // 2
                n2 = n - n1
                data = np.concatenate([
                    rng.normal(loc=-1.0 + loc, scale=0.5, size=n1),
                    rng.normal(loc= 1.0 + loc, scale=0.5, size=n2),
                ])

            else:
                data = rng.normal(loc=loc, scale=1.0, size=n)

            groups.append(data.tolist())

        return groups

    # -------------------------------------------------------------------------
    # Indeterminacy injection
    # -------------------------------------------------------------------------

    def induce_indeterminacy(
        self,
        neutro_array: NeutrosophicArray,
        delta: float,
        indeterminacy_type: str = "uniform",
        rng: Optional[np.random.Generator] = None,
    ) -> NeutrosophicArray:
        """
        Add controlled indeterminacy to a NeutrosophicArray.

        FIX 3: ``indices`` is now a Python ``set`` so ``i in indices`` is O(1)
        instead of O(n_indet) per element, cutting inner-loop cost from
        O(n * n_indet) to O(n).

        FIX 1/2 (inherited): accepts an explicit ``rng`` so the caller can
        pass a per-iteration Generator for reproducibility.
        """
        if rng is None:
            rng = self.rng

        n = len(neutro_array.data)
        n_indet = int(n * delta)
        if n_indet == 0:
            return neutro_array

        # FIX 3: convert to set for O(1) membership test
        indet_indices: set = set(rng.choice(n, size=n_indet, replace=False).tolist())

        t_mids  = [(x.T[0] + x.T[1]) / 2.0 for x in neutro_array.data]
        t_min   = min(t_mids)
        t_max   = max(t_mids)
        t_range = t_max - t_min if t_max > t_min else 1.0
        center  = (t_min + t_max) / 2.0

        modified = []
        for i, n_val in enumerate(neutro_array.data):
            if i not in indet_indices:          # O(1)
                modified.append(n_val)
                continue

            t_val = (n_val.T[0] + n_val.T[1]) / 2.0
            f_val = 1.0 - t_val

            if indeterminacy_type == "uniform":
                I = (0.0, 1.0)
            elif indeterminacy_type == "partial":
                I = (0.0, 0.5)
            elif indeterminacy_type == "mixed":
                I = (0.0, float(rng.uniform(0.2, 0.8)))
            elif indeterminacy_type == "data_dependent":
                dist_ratio = min(abs(t_val - center) / (t_range / 2.0), 1.0)
                I = (0.0, dist_ratio * 0.8 + 0.2)
            else:
                I = (0.0, 1.0)

            modified.append(NeutrosophicNumber((t_val, t_val), I, (f_val, f_val)))

        return NeutrosophicArray(modified)

    # -------------------------------------------------------------------------
    # Metric computation
    # -------------------------------------------------------------------------

    def compute_power_metrics(
        self,
        decisions: List[str],
        widths: List[float],
        effect_size: float,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute performance metrics from a list of simulation decisions.

        FIX 4: ``type1_error`` is now ``np.nan`` when ``effect_size > 0``
        (not 0.0) so that the output DataFrame clearly distinguishes
        "not measured" from "zero false-rejection rate".  Downstream code
        should use ``pd.notna()`` or ``dropna()`` before aggregating this field.
        """
        n = len(decisions)
        _empty = dict(
            power=0.0, type1_error=np.nan, rejection_rate=0.0,
            decision_stability=1.0, interval_width=0.0,
            indeterminate_rate=0.0, fail_to_reject_rate=1.0,
        )
        if n == 0:
            return _empty

        rejections     = sum(1 for d in decisions if d == "Reject H0")
        indeterminates = sum(1 for d in decisions if d == "Indeterminate Decision")
        fails          = sum(1 for d in decisions if d == "Fail to Reject H0")

        rejection_rate      = rejections     / n
        indeterminate_rate  = indeterminates / n
        fail_to_reject_rate = fails          / n
        decision_stability  = 1.0 - indeterminate_rate

        # FIX 4: type1_error is only defined under H0 (effect_size == 0)
        if effect_size == 0.0:
            type1_error = rejection_rate
            power       = 0.0
        else:
            type1_error = np.nan   # not applicable under H1
            power       = rejection_rate

        mean_width = float(np.mean(widths)) if widths else 0.0

        return dict(
            power=power,
            type1_error=type1_error,
            rejection_rate=rejection_rate,
            decision_stability=decision_stability,
            interval_width=mean_width,
            indeterminate_rate=indeterminate_rate,
            fail_to_reject_rate=fail_to_reject_rate,
        )

    def compute_relative_efficiency(
        self,
        modified_power: float,
        original_power: float,
        cap: float = _RE_CAP,
    ) -> float:
        """
        Relative efficiency = modified_power / original_power.

        FIX 5: ``inf`` values are capped at ``cap`` (default 3.0) before
        being stored in the DataFrame.  This prevents inf from propagating
        into aggregations such as .mean() and .std(), which would otherwise
        return inf or NaN for entire groups.

        FIX 6 (companion): the caller (run()) stores NaN for the 'original'
        variant row because RE is a comparison metric and does not apply to
        the baseline.
        """
        if np.isnan(modified_power) or np.isnan(original_power):
            return np.nan
        if original_power <= 0.0:
            return cap if modified_power > 0.0 else 1.0
        return min(modified_power / original_power, cap)

    # -------------------------------------------------------------------------
    # Single condition runner
    # -------------------------------------------------------------------------

    def run_single_condition(
        self,
        test_name: str,
        n: int,
        delta: float,
        distribution: str,
        effect_size: float,
        alpha: float = 0.05,
        indeterminacy_type: str = "uniform",
    ) -> Tuple[Dict, Dict]:
        """
        Run ``self.n_simulations`` iterations for one experimental condition.

        FIX 8 (critical): the submitted code returned raw lists
        ``(orig_decisions, orig_widths, mod_decisions, mod_widths)`` while
        ``run()`` immediately indexed the return value as a dict
        (``orig_metrics['power']``), causing a ``TypeError`` on every call.
        This method now returns ``(orig_metrics_dict, mod_metrics_dict)``.

        FIX 1 & 2: each iteration uses its own deterministic sub-Generator
        ``np.random.default_rng(self.random_seed * 10_000 + sim)`` so:
        - Results are fully reproducible given the same random_seed.
        - Iterations are statistically independent (no shared rng state).
        The multiplier 10_000 avoids seed collisions across typical sim ranges.
        """
        k = 2 if test_name == "mann_whitney" else 3

        orig_decisions: List[str]   = []
        orig_widths:    List[float] = []
        mod_decisions:  List[str]   = []
        mod_widths:     List[float] = []

        for sim in range(self.n_simulations):
            # FIX 1 & 2: independent, deterministic sub-stream per iteration
            iter_rng = np.random.default_rng(self.random_seed * 10_000 + sim)

            raw_groups = self.generate_data(
                n, distribution, effect_size, k=k, rng=iter_rng
            )

            neutro_groups = []
            for g in raw_groups:
                n_arr = neutrosophicate(g, indeterminacy_threshold=0.0)
                if delta > 0:
                    n_arr = self.induce_indeterminacy(
                        n_arr, delta, indeterminacy_type, rng=iter_rng
                    )
                neutro_groups.append(n_arr)

            # Dispatch to test functions
            if test_name == "kruskal_wallis":
                orig = kruskal_wallis_original(neutro_groups, alpha=alpha)
                mod  = kruskal_wallis_modified(neutro_groups, alpha=alpha)
            elif test_name == "mann_whitney":
                orig = mann_whitney_original(neutro_groups[0], neutro_groups[1], alpha=alpha)
                mod  = mann_whitney_modified(neutro_groups[0], neutro_groups[1], alpha=alpha)
            else:   # moods_median
                orig = moods_median_original(neutro_groups, alpha=alpha)
                mod  = moods_median_modified(neutro_groups, alpha=alpha)

            if orig is None or mod is None:
                continue

            # Both functions now expose a uniform 'p_interval' key (post-fix)
            piv_orig = orig.get("p_interval", (1.0, 1.0))
            piv_mod  = mod.get( "p_interval", (1.0, 1.0))

            orig_decisions.append(orig["decision_zone"])
            orig_widths.append(float(piv_orig[1] - piv_orig[0]))
            mod_decisions.append(mod["decision_zone"])
            mod_widths.append(float(piv_mod[1] - piv_mod[0]))

        orig_metrics = self.compute_power_metrics(
            orig_decisions, orig_widths, effect_size, alpha
        )
        mod_metrics = self.compute_power_metrics(
            mod_decisions, mod_widths, effect_size, alpha
        )
        return orig_metrics, mod_metrics

    # -------------------------------------------------------------------------
    # Full experiment runner
    # -------------------------------------------------------------------------

    def run(
        self,
        test_name: str,
        sample_sizes: List[int],
        indeterminacy_levels: List[float],
        distributions: List[str],
        effect_sizes: List[float],
        alpha: float = 0.05,
        indeterminacy_type: str = "uniform",
        progress_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Run the complete Monte Carlo experiment across all conditions.

        FIX 6: 'original' variant rows now store ``relative_efficiency=NaN``
        because RE is a comparison metric that only makes sense for the
        'modified' variant.  Storing 1.0 was misleading.

        FIX 7: removed the unused ``total_iterations`` variable.
        ``progress_callback`` is called with ``(current_condition, total_conditions)``
        as documented.
        """
        total_conditions = (
            len(sample_sizes) * len(indeterminacy_levels)
            * len(distributions) * len(effect_sizes)
        )
        current_condition = 0
        results = []

        for n in sample_sizes:
            for delta in indeterminacy_levels:
                for dist in distributions:
                    for effect in effect_sizes:
                        current_condition += 1

                        if progress_callback:
                            progress_callback(current_condition, total_conditions)

                        orig_metrics, mod_metrics = self.run_single_condition(
                            test_name, n, delta, dist, effect, alpha, indeterminacy_type
                        )

                        # FIX 5 (applied here): cap is already handled inside
                        # compute_relative_efficiency; no extra clamp needed.
                        re = self.compute_relative_efficiency(
                            mod_metrics["power"], orig_metrics["power"]
                        )

                        _base = dict(
                            test=test_name, n=n, delta=delta,
                            distribution=dist, effect_size=effect,
                        )

                        # FIX 6: original row — RE is not applicable (NaN)
                        results.append({
                            **_base,
                            "variant":              "original",
                            "power":                orig_metrics["power"],
                            "type1_error":          orig_metrics["type1_error"],
                            "rejection_rate":       orig_metrics["rejection_rate"],
                            "decision_stability":   orig_metrics["decision_stability"],
                            "interval_width":       orig_metrics["interval_width"],
                            "indeterminate_rate":   orig_metrics["indeterminate_rate"],
                            "fail_to_reject_rate":  orig_metrics["fail_to_reject_rate"],
                            "relative_efficiency":  np.nan,   # FIX 6
                        })

                        results.append({
                            **_base,
                            "variant":              "modified",
                            "power":                mod_metrics["power"],
                            "type1_error":          mod_metrics["type1_error"],
                            "rejection_rate":       mod_metrics["rejection_rate"],
                            "decision_stability":   mod_metrics["decision_stability"],
                            "interval_width":       mod_metrics["interval_width"],
                            "indeterminate_rate":   mod_metrics["indeterminate_rate"],
                            "fail_to_reject_rate":  mod_metrics["fail_to_reject_rate"],
                            "relative_efficiency":  re,
                        })

        df = pd.DataFrame(results)
        df.attrs.update(dict(
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
            alpha=alpha,
            indeterminacy_type=indeterminacy_type,
            test_name=test_name,
        ))
        return df

    # -------------------------------------------------------------------------
    # Post-processing helpers
    # -------------------------------------------------------------------------

    def summarize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate simulation results.

        FIX 9: uses ``ddof=0`` (population std) via a lambda instead of the
        default ``ddof=1``, so single-group std is 0.0 rather than NaN.
        NaN std values caused downstream plotting errors when only one
        condition was run.
        """
        agg_cols = [
            "power", "type1_error", "decision_stability",
            "interval_width", "indeterminate_rate", "relative_efficiency",
        ]
        agg_dict = {c: ["mean", lambda x: x.std(ddof=0)] for c in agg_cols}

        summary = (
            df.groupby(["test", "variant", "delta", "effect_size"])
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten multi-level column names: ('power', 'mean') -> 'power_mean'
        summary.columns = [
            "_".join(str(s) for s in col).strip("_")
            if isinstance(col, tuple) else col
            for col in summary.columns
        ]
        # Rename the lambda column <lambda> -> std
        summary.columns = [
            c.replace("<lambda>", "std") for c in summary.columns
        ]
        return summary

    def compute_power_curves(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[df["effect_size"] > 0]
            .groupby(["test", "variant", "n", "delta", "distribution", "effect_size"])["power"]
            .mean()
            .reset_index()
        )

    def compute_type1_error_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[df["effect_size"] == 0]
            .groupby(["test", "variant", "n", "delta", "distribution"])["type1_error"]
            .mean()
            .reset_index()
        )

    def compute_robustness_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sensitivity of test performance to increasing indeterminacy.

        FIX 10: ``np.polyfit`` with a single data point raises ``LinAlgError``.
        A guard is added: conditions with fewer than 2 delta levels are skipped.
        """
        robustness = []

        for test in df["test"].unique():
            for variant in df["variant"].unique():
                for n in df["n"].unique():
                    for dist in df["distribution"].unique():
                        for effect in df["effect_size"].unique():
                            subset = df[
                                (df["test"]         == test)    &
                                (df["variant"]      == variant) &
                                (df["n"]            == n)       &
                                (df["distribution"] == dist)    &
                                (df["effect_size"]  == effect)
                            ].sort_values("delta")

                            # FIX 10: need at least 2 points for a linear fit
                            if len(subset) < 2:
                                continue

                            deltas = subset["delta"].values

                            def _slope(col: str) -> float:
                                vals = subset[col].values
                                if np.all(np.isnan(vals)):
                                    return 0.0
                                # Replace NaN with 0 for slope estimation
                                vals = np.where(np.isnan(vals), 0.0, vals)
                                try:
                                    return float(np.polyfit(deltas, vals, 1)[0])
                                except np.linalg.LinAlgError:
                                    return 0.0

                            robustness.append(dict(
                                test=test,
                                variant=variant,
                                n=n,
                                distribution=dist,
                                effect_size=effect,
                                power_sensitivity=     _slope("power")     if effect > 0 else 0.0,
                                type1_sensitivity=     _slope("type1_error") if effect == 0 else 0.0,
                                stability_sensitivity= _slope("decision_stability"),
                                mean_interval_width=   float(subset["interval_width"].mean()),
                            ))

        return pd.DataFrame(robustness)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_full_simulation_comparison(
    n_simulations: int = 1000,
    random_seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full 3-test comparative simulation (Objective 3)."""

    sample_sizes         = [20, 50, 100, 200]
    indeterminacy_levels = [0.0, 0.1, 0.25, 0.4]
    distributions        = ["normal", "skewed", "heavy_tailed"]
    effect_sizes         = [0.0, 0.2, 0.5, 0.8, 1.0]

    simulator = MonteCarloSimulation(n_simulations=n_simulations, random_seed=random_seed)
    results   = {}

    for test_name in ["kruskal_wallis", "mann_whitney", "moods_median"]:
        print(f"\nRunning simulation for {test_name}...")

        df = simulator.run(
            test_name=test_name,
            sample_sizes=sample_sizes,
            indeterminacy_levels=indeterminacy_levels,
            distributions=distributions,
            effect_sizes=effect_sizes,
            alpha=0.05,
            indeterminacy_type="uniform",
            progress_callback=progress_callback,
        )

        results[test_name]                         = df
        results[f"{test_name}_summary"]            = simulator.summarize_results(df)
        results[f"{test_name}_power_curves"]       = simulator.compute_power_curves(df)
        results[f"{test_name}_robustness"]         = simulator.compute_robustness_indices(df)

    return results


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")

    print("=== 1. Reproducibility: same seed -> identical results ===")
    s1 = MonteCarloSimulation(n_simulations=3, random_seed=7)
    s2 = MonteCarloSimulation(n_simulations=3, random_seed=7)
    r1 = s1.generate_data(5, "normal", 0.5, k=3, rng=np.random.default_rng(7*10_000+0))
    r2 = s2.generate_data(5, "normal", 0.5, k=3, rng=np.random.default_rng(7*10_000+0))
    assert r1[0][0] == r2[0][0], "Same seed must produce identical data"
    print(f"  data[0][0] = {r1[0][0]:.6f} (both instances) ✓")

    print("\n=== 2. Iterations are independent (different seeds per sim) ===")
    s = MonteCarloSimulation(n_simulations=5, random_seed=42)
    vals = [s.generate_data(5,"normal",0.0,k=2,rng=np.random.default_rng(42*10_000+i))[0][0]
            for i in range(5)]
    assert len(set(f"{v:.8f}" for v in vals)) == 5, "All 5 iterations should differ"
    print(f"  5 iter values: {[f'{v:.4f}' for v in vals]} (all distinct) ✓")

    print("\n=== 3. induce_indeterminacy — O(1) membership ===")
    arr = NeutrosophicArray([
        NeutrosophicNumber((float(i)/99, float(i)/99), (0.0,0.0), (1-float(i)/99, 1-float(i)/99))
        for i in range(100)
    ])
    t0 = time.perf_counter()
    for _ in range(500):
        s.induce_indeterminacy(arr, 0.4, rng=np.random.default_rng(0))
    elapsed = time.perf_counter() - t0
    print(f"  500x n=100 delta=0.4: {elapsed:.3f}s ✓")

    print("\n=== 4. type1_error is NaN under H1 ===")
    m = s.compute_power_metrics(["Reject H0"]*8+["Fail to Reject H0"]*2, [0.1]*10, effect_size=0.5)
    assert np.isnan(m["type1_error"]), f"Expected NaN, got {m['type1_error']}"
    m0 = s.compute_power_metrics(["Reject H0"]*3+["Fail to Reject H0"]*7, [0.1]*10, effect_size=0.0)
    assert m0["type1_error"] == 0.3, f"Expected 0.3, got {m0['type1_error']}"
    print(f"  H1 type1_error={m['type1_error']}, H0 type1_error={m0['type1_error']} ✓")

    print("\n=== 5. compute_relative_efficiency caps inf ===")
    re = s.compute_relative_efficiency(0.8, 0.0)
    assert re == _RE_CAP, f"Expected {_RE_CAP}, got {re}"
    re2 = s.compute_relative_efficiency(0.5, 0.4)
    assert re2 == 0.5/0.4, f"Expected {0.5/0.4:.4f}, got {re2}"
    df_re = pd.DataFrame({"relative_efficiency": [1.0, re, 2.0]})
    assert np.isfinite(df_re["relative_efficiency"].mean()), "mean should be finite after cap"
    print(f"  RE(0.8, 0.0)={re} (capped at {_RE_CAP}) ✓")
    print(f"  RE(0.5, 0.4)={re2:.4f} ✓  mean is finite ✓")

    print("\n=== 6. Original variant row stores NaN for relative_efficiency ===")
    sim_small = MonteCarloSimulation(n_simulations=10, random_seed=0)
    df_small = sim_small.run(
        "kruskal_wallis", [20], [0.0], ["normal"], [0.0, 0.5], alpha=0.05
    )
    orig_re = df_small[df_small["variant"] == "original"]["relative_efficiency"]
    assert orig_re.isna().all(), "Original rows must have NaN relative_efficiency"
    mod_re  = df_small[df_small["variant"] == "modified"]["relative_efficiency"]
    assert mod_re.notna().all(), "Modified rows must have non-NaN relative_efficiency"
    print(f"  Original RE values: {orig_re.tolist()} (all NaN) ✓")

    print("\n=== 7. total_iterations is gone (no dead code) ===")
    import inspect
    src = inspect.getsource(MonteCarloSimulation.run)
    # Check only the executable lines of run(), not docstrings or comments
    run_lines = [l for l in src.split("\n")
                 if "total_iterations" in l and not l.strip().startswith(("#","'","\"","FIX","assert"))]
    assert len(run_lines) == 0, f"total_iterations still in live code: {run_lines}"
    print("  'total_iterations' not in run() source ✓")

    print("\n=== 8. run_single_condition returns (dict, dict) ===")
    o, m = sim_small.run_single_condition("kruskal_wallis", 20, 0.0, "normal", 0.5)
    assert isinstance(o, dict) and "power" in o, f"Expected dict with 'power', got {type(o)}"
    assert isinstance(m, dict) and "power" in m
    print(f"  orig_metrics['power']={o['power']:.3f}  mod_metrics['power']={m['power']:.3f} ✓")

    print("\n=== 9. summarize_results — no NaN std for single group ===")
    df_single = df_small.copy()
    summary = sim_small.summarize_results(df_single)
    std_cols = [c for c in summary.columns if "std" in c and "type1" not in c]
    for col in std_cols:
        assert not summary[col].isna().any(), f"NaN std found in column: {col}"
    print(f"  Checked {len(std_cols)} std columns — no NaN ✓")

    print("\n=== 10. compute_robustness_indices — single-delta guard ===")
    # Build a df with two delta levels so the polyfit guard can be tested properly
    df_two_deltas = sim_small.run(
        "kruskal_wallis", [20], [0.0, 0.25], ["normal"], [0.5], alpha=0.05
    )
    df_one_delta = df_two_deltas[df_two_deltas["delta"] == 0.0].copy()
    rob = sim_small.compute_robustness_indices(df_one_delta)
    assert len(rob) == 0, f"Expected 0 rows with single delta, got {len(rob)}"
    rob2 = sim_small.compute_robustness_indices(df_two_deltas)
    assert len(rob2) > 0, "Expected rows with two delta levels"
    print(f"  Single-delta: {len(rob)} rows (skipped safely) ✓")
    print(f"  Two-delta:    {len(rob2)} rows ✓")

    print("\nAll tests passed.")