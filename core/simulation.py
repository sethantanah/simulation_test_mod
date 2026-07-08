import numpy as np
import pandas as pd

from typing import Optional, Callable

from core.neutrosophic import (
    NeutrosophicNumber,
    NeutrosophicArray,
)

from core.tests.kruskal_wallis import (
    kruskal_wallis_original,
    kruskal_wallis_modified,
    build_neutrosophic_group,
)

from core.tests.mann_whitney import (
    mann_whitney_original,
    mann_whitney_modified,
)

from core.tests.moods_median import (
    moods_median_original,
    moods_median_modified,
)


# =============================================================================
# MONTE CARLO FRAMEWORK - FULLY CORRECTED
# =============================================================================

_RE_CAP = 3.0


class MonteCarloSimulation:

    def __init__(
        self,
        n_simulations: int = 1000,
        random_seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        
        self._supported_tests = {
            "kruskal_wallis",
            "mann_whitney", 
            "moods_median",
        }

    # =========================================================================
    # DETERMINISTIC SEED
    # =========================================================================

    def _make_seed(
        self,
        sim: int,
        n: int,
        delta: float,
        distribution: str,
        effect_size: float,
        test_name: str,
    ) -> int:
        """Create deterministic, reproducible seed for each simulation."""
        
        dist_code = {
            "normal": 11,
            "skewed": 23,
            "heavy_tailed": 37,
            "uniform": 41,
            "bimodal": 53,
        }.get(distribution, 97)

        test_code = {
            "kruskal_wallis": 101,
            "mann_whitney": 211,
            "moods_median": 307,
        }.get(test_name, 401)

        seed = (
            self.random_seed
            + sim * 100003
            + n * 1009
            + int(delta * 1000) * 917
            + int(effect_size * 1000) * 613
            + dist_code
            + test_code
        )

        return int(seed % (2**32 - 1))

    # =========================================================================
    # DATA GENERATION
    # =========================================================================

    def generate_data(
        self,
        n: int,
        distribution: str,
        effect_size: float,
        k: int,
        rng: np.random.Generator,
    ) -> list:
        """Generate synthetic data for k groups."""
        
        groups = []

        for i in range(k):
            loc = i * effect_size

            if distribution == "normal":
                data = rng.normal(loc=loc, scale=1.0, size=n)

            elif distribution == "skewed":
                # Lognormal centered to have mean approximately at loc
                data = rng.lognormal(mean=0.0, sigma=0.5, size=n) + loc - np.exp(0.125)

            elif distribution == "heavy_tailed":
                data = rng.standard_t(df=3, size=n) + loc

            elif distribution == "uniform":
                half = np.sqrt(3)
                data = rng.uniform(low=loc - half, high=loc + half, size=n)

            elif distribution == "bimodal":
                n1 = n // 2
                n2 = n - n1
                data = np.concatenate([
                    rng.normal(loc=-1.0 + loc, scale=0.5, size=n1),
                    rng.normal(loc=1.0 + loc, scale=0.5, size=n2),
                ])

            else:
                data = rng.normal(loc=loc, scale=1.0, size=n)

            groups.append(data.tolist())

        return groups

    # =========================================================================
    # CREATE NEUTROSOPHIC GROUPS - FIXED
    # =========================================================================
    
    @staticmethod
    def _create_crisp_neutrosophic(values: list) -> NeutrosophicArray:
        """
        Create neutrosophic array from crisp values.
        Uses build_neutrosophic_group with zero uncertainty and complement falsity.
        This is the SAME function used in the validated tests.
        """
        return build_neutrosophic_group(
            values, 
            uncertainty=0.0, 
            normalize=False, 
            falsity_mode="complement"
        )

    # =========================================================================
    # INDUCE INDETERMINACY - COMPLETELY REWRITTEN
    # =========================================================================

    def induce_indeterminacy(
        self,
        arr: NeutrosophicArray,
        delta: float,
        rng: np.random.Generator,
        indeterminacy_type: str = "uniform",
    ) -> NeutrosophicArray:
        """
        Add controlled indeterminacy to a FRACTION of observations.
        
        CRITICAL FIX: When delta=0, returns the array UNCHANGED.
        When delta>0, only modifies the indeterminacy component (I),
        leaving truth (T) and falsity (F) unchanged from the original.
        This preserves the ranking structure.
        """
        if delta <= 0:
            return arr

        n = len(arr.data)
        n_indet = int(np.round(delta * n))

        if n_indet <= 0:
            return arr

        # Select random subset for indeterminacy
        indices = set(
            rng.choice(n, size=n_indet, replace=False)
        )

        modified = []

        for i, x in enumerate(arr.data):
            if i not in indices:
                # Keep original neutrosophic number unchanged
                modified.append(x)
                continue

            # Determine uncertainty width
            if indeterminacy_type == "uniform":
                width = rng.uniform(0.5 * delta, min(1.0, 1.5 * delta))
            elif indeterminacy_type == "partial":
                width = rng.uniform(0.25 * delta, delta)
            elif indeterminacy_type == "mixed":
                width = rng.uniform(0.1, min(0.8, delta + 0.25))
            else:
                width = delta

            width = float(np.clip(width, 0.0, 1.0))

            # Only modify the indeterminacy component
            # Keep truth and falsity exactly as they were
            modified.append(
                NeutrosophicNumber(
                    (x.T[0], x.T[1]),     # Truth: unchanged
                    (0.0, width),          # Indeterminacy: new width
                    (x.F[0], x.F[1]),      # Falsity: unchanged
                )
            )

        return NeutrosophicArray(modified)

    # =========================================================================
    # DECISION ENCODING
    # =========================================================================

    @staticmethod
    def encode_rejection(decision: str) -> int:
        """Encode rejection decision."""
        return int("reject" in decision.lower() and "fail" not in decision.lower())

    @staticmethod
    def encode_indeterminate(decision: str) -> int:
        """Encode indeterminate decision."""
        return int("indeterminate" in decision.lower())

    # =========================================================================
    # DECISION STABILITY
    # =========================================================================

    @staticmethod
    def compute_decision_stability(decisions) -> float:
        """Compute proportion of decisive (non-indeterminate) outcomes."""
        if len(decisions) == 0:
            return np.nan
        decisions = np.asarray(decisions)
        return float(np.mean(decisions != "Indeterminate Decision"))

    # =========================================================================
    # RELATIVE EFFICIENCY
    # =========================================================================

    @staticmethod
    def relative_efficiency(mod_power: float, orig_power: float) -> float:
        """Compute relative efficiency of modified vs original test."""
        if np.isnan(mod_power) or np.isnan(orig_power):
            return np.nan
        if orig_power <= 0:
            if mod_power <= 0:
                return 1.0
            return _RE_CAP
        return float(min(mod_power / orig_power, _RE_CAP))

    # =========================================================================
    # TEST DISPATCH
    # =========================================================================

    def dispatch_test(
        self,
        test_name: str,
        neutro_groups: list,
        alpha: float,
    ) -> tuple:
        """Dispatch to appropriate test function."""
        
        if test_name not in self._supported_tests:
            raise ValueError(f"Unsupported test: {test_name}")

        if test_name == "kruskal_wallis":
            original = kruskal_wallis_original(neutro_groups, alpha=alpha)
            modified = kruskal_wallis_modified(neutro_groups, alpha=alpha)

        elif test_name == "mann_whitney":
            if len(neutro_groups) != 2:
                raise ValueError("Mann-Whitney requires exactly 2 groups")
            original = mann_whitney_original(
                neutro_groups[0], neutro_groups[1], alpha=alpha
            )
            modified = mann_whitney_modified(
                neutro_groups[0], neutro_groups[1], alpha=alpha
            )

        elif test_name == "moods_median":
            original = moods_median_original(neutro_groups, alpha=alpha)
            modified = moods_median_modified(neutro_groups, alpha=alpha)

        return original, modified

    # =========================================================================
    # VALIDATION - CHECK CLASSICAL EQUIVALENCE
    # =========================================================================
    
    def validate_classical_equivalence(self, test_name: str = "kruskal_wallis") -> bool:
        """
        Verify that the simulation reproduces classical Type I error rates
        when delta=0. This must pass before running full simulations.
        """
        from scipy.stats import kruskal
        
        n_val = 50
        n_sims_val = 500
        alpha = 0.05
        
        rejections_classical = 0
        rejections_neutro_orig = 0
        rejections_neutro_mod = 0
        
        for sim in range(n_sims_val):
            seed = self._make_seed(sim, n_val, 0.0, "normal", 0.0, test_name)
            rng = np.random.default_rng(seed)
            
            # Generate data
            raw_groups = self.generate_data(n_val, "normal", 0.0, 3, rng)
            
            # Classical Kruskal-Wallis
            _, p_class = kruskal(*raw_groups)
            if p_class < alpha:
                rejections_classical += 1
            
            # Neutrosophic with delta=0 (no indeterminacy induced)
            neutro_groups = [
                self._create_crisp_neutrosophic(g) for g in raw_groups
            ]
            
            r_orig = kruskal_wallis_original(neutro_groups, alpha=alpha)
            r_mod = kruskal_wallis_modified(neutro_groups, alpha=alpha)
            
            if r_orig["decision_zone"] == "Reject H0":
                rejections_neutro_orig += 1
            if r_mod["decision_zone"] == "Reject H0":
                rejections_neutro_mod += 1
        
        classical_rate = rejections_classical / n_sims_val
        neutro_orig_rate = rejections_neutro_orig / n_sims_val
        neutro_mod_rate = rejections_neutro_mod / n_sims_val
        
        print(f"\n  Classical equivalence validation (n={n_val}, sims={n_sims_val}):")
        print(f"    Classical Kruskal-Wallis: {classical_rate:.4f}")
        print(f"    Neutrosophic Original:    {neutro_orig_rate:.4f}")
        print(f"    Neutrosophic Modified:    {neutro_mod_rate:.4f}")
        
        # All three should be within ~2 standard errors of each other
        se = np.sqrt(alpha * (1 - alpha) / n_sims_val)
        tolerance = 3 * se  # 3 standard errors
        
        if abs(classical_rate - neutro_orig_rate) > tolerance:
            print(f"    WARNING: Original variant differs from classical!")
            return False
        
        if abs(classical_rate - neutro_mod_rate) > tolerance:
            print(f"    WARNING: Modified variant differs from classical!")
            return False
        
        print(f"    ✓ Classical equivalence verified")
        return True

    # =========================================================================
    # SINGLE CONDITION
    # =========================================================================

    def run_single_condition(
        self,
        test_name: str,
        n: int,
        delta: float,
        distribution: str,
        effect_size: float,
        alpha: float,
        indeterminacy_type: str,
    ) -> list:
        """Run simulations for a single parameter combination."""
        
        if test_name not in self._supported_tests:
            raise ValueError(f"Unsupported test: {test_name}")
        
        k = 2 if test_name == "mann_whitney" else 3
        rows = []

        for sim in range(self.n_simulations):
            seed = self._make_seed(
                sim=sim, n=n, delta=delta,
                distribution=distribution, effect_size=effect_size,
                test_name=test_name,
            )
            rng = np.random.default_rng(seed)

            # Generate raw data
            raw_groups = self.generate_data(
                n=n, distribution=distribution,
                effect_size=effect_size, k=k, rng=rng,
            )

            # Create neutrosophic groups using the SAME method as validated tests
            neutro_groups = []
            for g in raw_groups:
                # Start with crisp neutrosophic numbers
                arr = self._create_crisp_neutrosophic(g)
                
                # Add controlled indeterminacy (no-op when delta=0)
                arr = self.induce_indeterminacy(
                    arr=arr, delta=delta, rng=rng,
                    indeterminacy_type=indeterminacy_type,
                )
                neutro_groups.append(arr)

            # Apply tests
            original, modified = self.dispatch_test(
                test_name=test_name,
                neutro_groups=neutro_groups,
                alpha=alpha,
            )

            for variant, res in [("original", original), ("modified", modified)]:
                decision = res.get("decision_zone", 
                                    res.get("overall_decision", "Unknown"))
                reject = self.encode_rejection(decision)
                indeterminate = self.encode_indeterminate(decision)

                pval = res.get("p_value")
                if pval is None:
                    if "p_interval" in res:
                        pval = (res["p_interval"][0] + res["p_interval"][1]) / 2.0
                    else:
                        pval = res.get("p_T_modified", res.get("p_T", np.nan))

                p_interval = res.get("p_interval", (pval, pval))
                interval_width = float(p_interval[1] - p_interval[0])

                rows.append({
                    "test": test_name,
                    "variant": variant,
                    "n": n,
                    "delta": delta,
                    "distribution": distribution,
                    "effect_size": effect_size,
                    "simulation": sim,
                    "decision": decision,
                    "reject": reject,
                    "indeterminate": indeterminate,
                    "p_value": float(pval),
                    "interval_width": interval_width,
                })

        return rows

    # =========================================================================
    # FULL RUN
    # =========================================================================

    def run(
        self,
        test_name: str,
        sample_sizes: list,
        indeterminacy_levels: list,
        distributions: list,
        effect_sizes: list,
        alpha: float = 0.05,
        indeterminacy_type: str = "uniform",
        progress_callback: Optional[Callable] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Run full Monte Carlo simulation across all parameter combinations."""
        
        if test_name not in self._supported_tests:
            raise ValueError(
                f"Unsupported test: {test_name}. "
                f"Supported tests: {self._supported_tests}"
            )
        
        # Validate classical equivalence before running
        if validate:
            print("\nValidating classical equivalence...")
            if not self.validate_classical_equivalence(test_name):
                raise RuntimeError(
                    "Classical equivalence check failed. "
                    "Simulation cannot proceed with broken tests."
                )
        
        rows = []
        total = (
            len(sample_sizes)
            * len(indeterminacy_levels)
            * len(distributions)
            * len(effect_sizes)
        )
        step = 0

        print(f"\nRunning {total} conditions × {self.n_simulations} simulations each...")
        
        for n in sample_sizes:
            for delta in indeterminacy_levels:
                for dist in distributions:
                    for effect in effect_sizes:
                        step += 1
                        if progress_callback:
                            progress_callback(step, total)

                        cond_rows = self.run_single_condition(
                            test_name=test_name,
                            n=n,
                            delta=delta,
                            distribution=dist,
                            effect_size=effect,
                            alpha=alpha,
                            indeterminacy_type=indeterminacy_type,
                        )
                        rows.extend(cond_rows)

        raw_df = pd.DataFrame(rows)
        return self.summarize_results(raw_df)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summarize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate simulation results into summary statistics."""
        
        grouped = df.groupby([
            "test", "variant", "n", "delta",
            "distribution", "effect_size",
        ])

        out = grouped.agg(
            rejection_rate=("reject", "mean"),
            rejection_std=("reject", lambda x: float(np.std(x, ddof=0))),
            indeterminate_rate=("indeterminate", "mean"),
            decision_stability=("decision", self.compute_decision_stability),
            interval_width_mean=("interval_width", "mean"),
            interval_width_std=("interval_width", lambda x: float(np.std(x, ddof=0))),
            p_value_mean=("p_value", "mean"),
            p_value_std=("p_value", lambda x: float(np.std(x, ddof=0))),
        ).reset_index()

        # Fail-to-reject rate
        out["fail_to_reject_rate"] = (
            1.0 - out["rejection_rate"] - out["indeterminate_rate"]
        )

        # Power and Type I error
        out["power"] = np.where(
            out["effect_size"] > 0,
            out["rejection_rate"],
            np.nan,
        )
        out["type1_error"] = np.where(
            out["effect_size"] == 0,
            out["rejection_rate"],
            np.nan,
        )

        # Relative efficiency
        out["relative_efficiency"] = np.nan

        for idx, row in out.iterrows():
            if row["variant"] != "modified":
                continue
            if row["effect_size"] <= 0:
                continue

            mask = (
                (out["test"] == row["test"])
                & (out["n"] == row["n"])
                & (out["delta"] == row["delta"])
                & (out["distribution"] == row["distribution"])
                & (out["effect_size"] == row["effect_size"])
                & (out["variant"] == "original")
            )

            orig = out.loc[mask]
            if len(orig) == 0:
                continue

            orig_power = orig.iloc[0]["power"]
            mod_power = row["power"]

            out.at[idx, "relative_efficiency"] = self.relative_efficiency(
                mod_power, orig_power
            )

        return out