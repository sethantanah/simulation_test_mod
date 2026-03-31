import numpy as np
import pandas as pd
from scipy.stats import rankdata


class NeutrosophicNumber:
    """
    Represents a neutrosophic number as a triple (T, I, F)
    where T=Truth, I=Indeterminacy, F=Falsehood.
    Each component is an interval [lower, upper].

    Standard encoding for a crisp value x in [min, max]:
        T = (normalized_x, normalized_x)
        F = (1 - normalized_x, 1 - normalized_x)
        I = (0, indeterminacy_band_width)   [0 when no uncertainty]

    Classical special case: when all I = (0,0), neutrosophic tests
    reduce exactly to their classical counterparts.
    """

    def __init__(self, T: tuple, I: tuple, F: tuple):
        self.T = tuple(float(x) for x in T)
        self.I = tuple(float(x) for x in I)
        self.F = tuple(float(x) for x in F)

    # ------------------------------------------------------------------
    # Arithmetic — full interval arithmetic across all three components
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = NeutrosophicNumber((other, other), (0.0, 0.0), (0.0, 0.0))
        return NeutrosophicNumber(
            (self.T[0] + other.T[0], self.T[1] + other.T[1]),
            (self.I[0] + other.I[0], self.I[1] + other.I[1]),
            (self.F[0] + other.F[0], self.F[1] + other.F[1]),
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = NeutrosophicNumber((other, other), (0.0, 0.0), (0.0, 0.0))
        return NeutrosophicNumber(
            (self.T[0] - other.T[1], self.T[1] - other.T[0]),
            (self.I[0] - other.I[1], self.I[1] - other.I[0]),
            (self.F[0] - other.F[1], self.F[1] - other.F[0]),
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # FIX: a crisp scalar has I=(0,0), F=(0,0) — not (s,s) for all components
            other = NeutrosophicNumber((other, other), (0.0, 0.0), (0.0, 0.0))

        def mul_interval(i1, i2):
            vals = [i1[0] * i2[0], i1[0] * i2[1], i1[1] * i2[0], i1[1] * i2[1]]
            return (min(vals), max(vals))

        return NeutrosophicNumber(
            mul_interval(self.T, other.T),
            mul_interval(self.I, other.I),
            mul_interval(self.F, other.F),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero scalar.")
            # FIX: a crisp scalar has I=(0,0), F=(0,0) — not (s,s) for all components.
            # Previously `other = NeutrosophicNumber((s,s),(s,s),(s,s))` which wrongly
            # set the scalar's indeterminacy to s, inflating I in every division result.
            other = NeutrosophicNumber((other, other), (0.0, 0.0), (0.0, 0.0))

        def div_interval(i1, i2):
            if i2[0] <= 0 <= i2[1]:
                raise ZeroDivisionError("Division by zero-containing interval.")
            i2_inv = (1.0 / i2[1], 1.0 / i2[0])
            vals = [i1[0] * i2_inv[0], i1[0] * i2_inv[1],
                    i1[1] * i2_inv[0], i1[1] * i2_inv[1]]
            return (min(vals), max(vals))

        # Special case: when the denominator's I or F component is (0,0) — which
        # happens for crisp scalars — we cannot call div_interval (it would try
        # to invert a zero interval). Instead, scale the numerator's component
        # by dividing through by the denominator's T-midpoint (the crisp value).
        # This is mathematically correct: dividing a neutrosophic number by a
        # crisp scalar s scales each component by 1/s independently.
        def safe_div(i1, i2, fallback_scalar):
            if i2 == (0.0, 0.0):
                if fallback_scalar == 0.0:
                    if i1 == (0.0, 0.0):
                        return (0.0, 0.0)
                    raise ZeroDivisionError(f"Cannot divide {i1} by zero scalar.")
                return (i1[0] / fallback_scalar, i1[1] / fallback_scalar)
            return div_interval(i1, i2)

        # midpoint of denominator T used as the fallback scalar for I and F
        denom_scalar = (other.T[0] + other.T[1]) / 2.0

        return NeutrosophicNumber(
            div_interval(self.T, other.T),
            safe_div(self.I, other.I, denom_scalar),
            safe_div(self.F, other.F, denom_scalar),
        )

    def __repr__(self):
        return (
            f"N(T=[{self.T[0]:.4g}, {self.T[1]:.4g}], "
            f"I=[{self.I[0]:.4g}, {self.I[1]:.4g}], "
            f"F=[{self.F[0]:.4g}, {self.F[1]:.4g}])"
        )

    def __str__(self):
        return self.__repr__()

    # ------------------------------------------------------------------
    # Derived scalar properties
    # ------------------------------------------------------------------

    def score(self) -> float:
        """S = T_mean - F_mean. Higher = more true."""
        return ((self.T[0] + self.T[1]) / 2.0) - ((self.F[0] + self.F[1]) / 2.0)

    def accuracy(self) -> float:
        """A = T_mean + F_mean. Higher = less indeterminate."""
        return ((self.T[0] + self.T[1]) / 2.0) + ((self.F[0] + self.F[1]) / 2.0)

    def defuzzify(self) -> float:
        """
        Crisp representative value via de-neutrosophication.
        Formula: (2 + T_m - I_m - F_m) / 3, clamped to [0, 1].

        FIX: added clamp to guarantee output stays in [0, 1] for all
        valid neutrosophic inputs.
        """
        t_m = (self.T[0] + self.T[1]) / 2.0
        i_m = (self.I[0] + self.I[1]) / 2.0
        f_m = (self.F[0] + self.F[1]) / 2.0
        raw = (2.0 + t_m - i_m - f_m) / 3.0
        return float(max(0.0, min(1.0, raw)))

    def is_indeterminate(self, threshold: float = 0.01) -> bool:
        """True if the width of the I-interval exceeds threshold."""
        return (self.I[1] - self.I[0]) > threshold

    def to_dict(self) -> dict:
        return {
            "T_lower": self.T[0], "T_upper": self.T[1],
            "I_lower": self.I[0], "I_upper": self.I[1],
            "F_lower": self.F[0], "F_upper": self.F[1],
        }


# ---------------------------------------------------------------------------
# NeutrosophicArray
# ---------------------------------------------------------------------------

class NeutrosophicArray:
    """A sequence of NeutrosophicNumber objects with array-level operations."""

    def __init__(self, data: list):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # FIX: explicit __iter__ so zip(), list(), enumerate(), etc. all work correctly
    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"NeutrosophicArray([{', '.join(repr(n) for n in self.data[:3])}{'...' if len(self.data) > 3 else ''}])"

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self) -> "NeutrosophicArray":
        """
        Compute neutrosophic ranks.

        FIX (was: ranking I-midpoints and F-midpoints independently, which
        has no meaning in KW/MWU tests):

        The DATA rank of each observation is determined solely by the
        T-component midpoint (the truth value = the observed measurement).
        The I and F components express *uncertainty about that observation*,
        not independent variables to be ranked.

        Each rank is returned as a NeutrosophicNumber where:
          - T component = the classical rank (point interval, width=0)
          - I component = (0, 0)  [modified KW stretches this separately]
          - F component = (0, 0)

        The modified Kruskal-Wallis test builds interval-valued ranks on top
        of these point ranks by widening the T-interval based on each
        observation's I-width. That logic lives in the modified KW method,
        not here, keeping this method consistent with the original test spec.

        Ties are handled with the standard average-rank method (method='average').
        """
        t_mids = [(n.T[0] + n.T[1]) / 2.0 for n in self.data]
        t_ranks = rankdata(t_mids, method="average")

        ranked = []
        for r in t_ranks:
            r = float(r)
            ranked.append(NeutrosophicNumber((r, r), (0.0, 0.0), (0.0, 0.0)))

        return NeutrosophicArray(ranked)

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def neutrosophic_median(self) -> NeutrosophicNumber:
        """Component-wise median of lower and upper bounds."""
        return NeutrosophicNumber(
            (np.median([n.T[0] for n in self.data]),
             np.median([n.T[1] for n in self.data])),
            (np.median([n.I[0] for n in self.data]),
             np.median([n.I[1] for n in self.data])),
            (np.median([n.F[0] for n in self.data]),
             np.median([n.F[1] for n in self.data])),
        )

    def summary_stats(self) -> dict:
        """
        Returns T/I/F means, variances (ddof=1), and global min/max.
        Each entry is a dict with keys: mean, var, min, max.
        """
        def calc_stats(get_comp):
            lowers = [get_comp(n)[0] for n in self.data]
            uppers = [get_comp(n)[1] for n in self.data]
            n = len(lowers)
            return {
                "mean":  (float(np.mean(lowers)), float(np.mean(uppers))),
                "var":   (float(np.var(lowers, ddof=1)) if n > 1 else 0.0,
                          float(np.var(uppers, ddof=1)) if n > 1 else 0.0),
                "min":   float(np.min(lowers)),
                "max":   float(np.max(uppers)),
            }

        return {
            "T": calc_stats(lambda n: n.T),
            "I": calc_stats(lambda n: n.I),
            "F": calc_stats(lambda n: n.F),
        }


# ---------------------------------------------------------------------------
# neutrosophicate()
# ---------------------------------------------------------------------------

def neutrosophicate(
    data: list,
    indeterminacy_threshold: float = 0.1,
) -> NeutrosophicArray:
    """
    Convert a list of crisp (or partially missing) values into a
    NeutrosophicArray using the standard [T, I, F] encoding.

    Encoding rules
    --------------
    For a valid crisp value x in dataset with range [min, max]:

        normalized = (x - min) / range          ∈ [0, 1]

        T = (normalized, normalized)             — truth  = relative magnitude
        F = (1 - normalized, 1 - normalized)     — falsehood = complement
        I = (0, band_width)                      — indeterminacy band

    FIX 1 — F was always (0,0): F must be the complement of T so that the
    classical special case holds: when I=(0,0), score() = T_m - F_m = 2*norm-1
    and the ranking on T-midpoints reproduces classical ordinal ranking.

    FIX 2 — I-interval for missing values was in raw data units (0, global_range).
    It is now always in normalized [0, 1] space: I = (0.0, 1.0).

    Indeterminacy band width
    ------------------------
    - Missing / NaN     → T=(0,0), I=(0,1), F=(0,0)  [fully indeterminate]
    - Near boundary     → I = (0, 0.1)               [slight uncertainty]
    - Interior value    → I = (0, 0)                 [fully determinate]

    Classical special case validation
    ----------------------------------
    When indeterminacy_threshold = 0 (or all values are interior):
        Every observation gets I=(0,0).
        rank() ranks on T-midpoints = normalized values.
        This reproduces classical ordinal ranking → KW H = classical H. ✓
    """
    valid_data = [x for x in data if not pd.isna(x)]

    if len(valid_data) == 0:
        # All missing: return fully indeterminate array
        return NeutrosophicArray(
            [NeutrosophicNumber((0.0, 0.0), (0.0, 1.0), (0.0, 0.0))
             for _ in data]
        )

    global_min = float(min(valid_data))
    global_max = float(max(valid_data))
    global_range = global_max - global_min if global_max > global_min else 1.0

    result = []
    for val in data:
        if pd.isna(val):
            # FIX 2: I-width is 1.0 in normalized [0,1] space, not global_range
            result.append(
                NeutrosophicNumber((0.0, 0.0), (0.0, 1.0), (0.0, 0.0))
            )
        else:
            normalized = (float(val) - global_min) / global_range

            # FIX 1: F = complement of T, not (0,0)
            T = (normalized, normalized)
            F = (1.0 - normalized, 1.0 - normalized)

            # Indeterminacy: widen for values near the range boundary
            bound_dist = min(normalized, 1.0 - normalized)
            if bound_dist < indeterminacy_threshold:
                # Proportional to how close to the boundary
                i_width = indeterminacy_threshold * (1.0 - bound_dist / indeterminacy_threshold) * 0.1
                I = (0.0, round(i_width, 6))
            else:
                I = (0.0, 0.0)

            result.append(NeutrosophicNumber(T, I, F))

    return NeutrosophicArray(result)


# ---------------------------------------------------------------------------
# Quick self-test  (run with: python neutrosophic.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NeutrosophicNumber arithmetic tests ===\n")

    a = NeutrosophicNumber((0.6, 0.8), (0.1, 0.2), (0.1, 0.3))
    b = NeutrosophicNumber((0.4, 0.5), (0.0, 0.1), (0.3, 0.5))

    print(f"a         = {a}")
    print(f"b         = {b}")
    print(f"a + b     = {a + b}")
    print(f"a - b     = {a - b}")
    print(f"a * b     = {a * b}")
    print(f"a / 2     = {a / 2}")          # scalar division — I should stay narrow
    print(f"a.score() = {a.score():.4f}")
    print(f"a.defuzz  = {a.defuzzify():.4f}")

    print("\n=== Classical special-case validation ===\n")
    # All interior values, threshold=0 → all I=(0,0) → crisp array
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    arr = neutrosophicate(data, indeterminacy_threshold=0.0)
    for i, n in enumerate(arr):
        assert n.I == (0.0, 0.0), f"Expected I=(0,0) for interior value, got {n.I}"
    print("All I=(0,0) when threshold=0 ✓")

    ranks = arr.rank()
    t_mids = [(n.T[0] + n.T[1]) / 2.0 for n in arr]
    classical_ranks = rankdata(t_mids, method="average")
    neutro_ranks = [(r.T[0] + r.T[1]) / 2.0 for r in ranks]
    assert list(neutro_ranks) == list(classical_ranks), \
        f"Rank mismatch:\n  neutro={neutro_ranks}\n  classical={list(classical_ranks)}"
    print("Neutrosophic ranks == classical ranks when I=(0,0) ✓")

    print("\n=== Missing value test ===\n")
    data_with_nan = [1.0, float("nan"), 3.0]
    arr2 = neutrosophicate(data_with_nan)
    nan_num = arr2[1]
    assert nan_num.I == (0.0, 1.0), f"Missing value I should be (0,1), got {nan_num.I}"
    assert nan_num.T == (0.0, 0.0), f"Missing value T should be (0,0), got {nan_num.T}"
    print(f"Missing value → {nan_num} ✓")

    print("\n=== F-complement test ===\n")
    arr3 = neutrosophicate([0.0, 0.5, 1.0], indeterminacy_threshold=0.0)
    for n in arr3:
        t_m = (n.T[0] + n.T[1]) / 2.0
        f_m = (n.F[0] + n.F[1]) / 2.0
        assert abs(t_m + f_m - 1.0) < 1e-9, \
            f"T_mean + F_mean should = 1.0, got {t_m + f_m}"
    print("T_mean + F_mean = 1.0 for all crisp values ✓")

    print("\n=== defuzzify clamp test ===\n")
    extreme = NeutrosophicNumber((1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    assert 0.0 <= extreme.defuzzify() <= 1.0, "defuzzify must be in [0,1]"
    print(f"defuzzify extreme = {extreme.defuzzify():.4f} ✓")

    print("\n=== NeutrosophicArray.__iter__ test ===\n")
    items = list(arr)
    assert len(items) == len(arr), "__iter__ length mismatch"
    zipped = list(zip(arr, arr))
    assert len(zipped) == len(arr), "zip() failed — __iter__ missing?"
    print("__iter__, list(), zip() all work ✓")

    print("\nAll tests passed.")