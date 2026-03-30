import numpy as np
import pandas as pd
from scipy.stats import rankdata

class NeutrosophicNumber:
    """
    Represents a neutrosophic number as a triple (T, I, F)
    where T=Truth, I=Indeterminacy, F=Falsehood.
    Each component is an interval [lower, upper].
    """
    def __init__(self, T: tuple, I: tuple, F: tuple):
        self.T = tuple(float(x) for x in T)
        self.I = tuple(float(x) for x in I)
        self.F = tuple(float(x) for x in F)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = NeutrosophicNumber((other, other), (0, 0), (0, 0))
        return NeutrosophicNumber(
            (self.T[0] + other.T[0], self.T[1] + other.T[1]),
            (self.I[0] + other.I[0], self.I[1] + other.I[1]),
            (self.F[0] + other.F[0], self.F[1] + other.F[1])
        )

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = NeutrosophicNumber((other, other), (0, 0), (0, 0))
        return NeutrosophicNumber(
            (self.T[0] - other.T[1], self.T[1] - other.T[0]),
            (self.I[0] - other.I[1], self.I[1] - other.I[0]),
            (self.F[0] - other.F[1], self.F[1] - other.F[0])
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = NeutrosophicNumber((other, other), (other, other), (other, other))
        def mul_interval(i1, i2):
            vals = [i1[0]*i2[0], i1[0]*i2[1], i1[1]*i2[0], i1[1]*i2[1]]
            return (min(vals), max(vals))
        return NeutrosophicNumber(
            mul_interval(self.T, other.T),
            mul_interval(self.I, other.I),
            mul_interval(self.F, other.F)
        )

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError()
            other = NeutrosophicNumber((other, other), (other, other), (other, other))
        def div_interval(i1, i2):
            if i2[0] <= 0 <= i2[1]: raise ZeroDivisionError("Division by zero interval")
            i2_inv = (1.0 / i2[1], 1.0 / i2[0])
            vals = [i1[0]*i2_inv[0], i1[0]*i2_inv[1], i1[1]*i2_inv[0], i1[1]*i2_inv[1]]
            return (min(vals), max(vals))
        return NeutrosophicNumber(
            div_interval(self.T, other.T),
            div_interval(self.I, other.I),
            div_interval(self.F, other.F)
        )

    def __repr__(self):
        return f"N([{self.T[0]:.4g}, {self.T[1]:.4g}], [{self.I[0]:.4g}, {self.I[1]:.4g}], [{self.F[0]:.4g}, {self.F[1]:.4g}])"

    def __str__(self): return self.__repr__()

    def score(self) -> float:
        return ((self.T[0]+self.T[1])/2) - ((self.F[0]+self.F[1])/2)

    def accuracy(self) -> float:
        return ((self.T[0]+self.T[1])/2) + ((self.F[0]+self.F[1])/2)

    def defuzzify(self) -> float:
        tm = (self.T[0] + self.T[1]) / 2.0
        im = (self.I[0] + self.I[1]) / 2.0
        fm = (self.F[0] + self.F[1]) / 2.0
        return (2 + tm - im - fm) / 3.0

    def is_indeterminate(self, threshold: float = 0.01) -> bool:
        return (self.I[1] - self.I[0]) > threshold

    def to_dict(self) -> dict:
        return {
            "T_lower": self.T[0], "T_upper": self.T[1],
            "I_lower": self.I[0], "I_upper": self.I[1],
            "F_lower": self.F[0], "F_upper": self.F[1]
        }

class NeutrosophicArray:
    def __init__(self, data: list):
        self.data = data
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

    def rank(self) -> 'NeutrosophicArray':
        # Classical neutrosophic ranks are independent on T, I, F
        t_mids = [(n.T[0] + n.T[1]) / 2.0 for n in self.data]
        i_mids = [(n.I[0] + n.I[1]) / 2.0 for n in self.data]
        f_mids = [(n.F[0] + n.F[1]) / 2.0 for n in self.data]
        
        t_ranks = rankdata(t_mids, method='average')
        i_ranks = rankdata(i_mids, method='average')
        f_ranks = rankdata(f_mids, method='average')
        
        res = []
        for i in range(len(self.data)):
            rT = float(t_ranks[i])
            rI = float(i_ranks[i])
            rF = float(f_ranks[i])
            
            # Create a neutrosophic number for the rank
            # By default it's an exact rank (interval length 0), but the Kruskal-Wallis
            # modification will stretch this rank into an interval based on indeterminacy width.
            res.append(NeutrosophicNumber((rT, rT), (rI, rI), (rF, rF)))
            
        return NeutrosophicArray(res)

    def neutrosophic_median(self) -> NeutrosophicNumber:
        return NeutrosophicNumber(
            (np.median([n.T[0] for n in self.data]), np.median([n.T[1] for n in self.data])),
            (np.median([n.I[0] for n in self.data]), np.median([n.I[1] for n in self.data])),
            (np.median([n.F[0] for n in self.data]), np.median([n.F[1] for n in self.data]))
        )

    def summary_stats(self) -> dict:
        def calc_stats(comp_func):
            lowers = [comp_func(n)[0] for n in self.data]
            uppers = [comp_func(n)[1] for n in self.data]
            return {
                "mean": (np.mean(lowers), np.mean(uppers)),
                "var": (np.var(lowers, ddof=1) if len(lowers) > 1 else 0.0, np.var(uppers, ddof=1) if len(uppers) > 1 else 0.0),
                "min": np.min(lowers), "max": np.max(uppers)
            }
        return {
            "T": calc_stats(lambda n: n.T),
            "I": calc_stats(lambda n: n.I),
            "F": calc_stats(lambda n: n.F)
        }

def neutrosophicate(data: list, indeterminacy_threshold: float = 0.1) -> NeutrosophicArray:
    """
    Takes crisp or partially missing data into NeutrosophicArray
    Missing values -> indeterminate with wide I interval
    Values within threshold of boundary -> broadened I interval
    """
    res = []
    valid_data = [x for x in data if not pd.isna(x)]
    if len(valid_data) == 0:
        return NeutrosophicArray([NeutrosophicNumber((0,0), (0,1), (0,0)) for _ in data])
        
    global_min, global_max = min(valid_data), max(valid_data)
    global_range = global_max - global_min if global_max > global_min else 1.0
    
    for val in data:
        if pd.isna(val):
            res.append(NeutrosophicNumber((0, 0), (0, global_range), (0, 0)))
        else:
            T = (float(val), float(val))
            F = (0.0, 0.0) 
            
            normalized = (val - global_min) / global_range
            bound_dist = min(normalized, 1.0 - normalized)
            
            if bound_dist < indeterminacy_threshold:
                I = (0.0, global_range * 0.1)
            else:
                I = (0.0, 0.0)
                
            res.append(NeutrosophicNumber(T, I, F))
    return NeutrosophicArray(res)
