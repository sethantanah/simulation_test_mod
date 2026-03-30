import numpy as np
import pandas as pd
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray, neutrosophicate
from core.tests.kruskal_wallis import kruskal_wallis_original, kruskal_wallis_modified
from core.tests.mann_whitney import mann_whitney_original, mann_whitney_modified
from core.tests.moods_median import moods_median_original, moods_median_modified

class MonteCarloSimulation:
    def __init__(self, n_simulations=1000, random_seed=42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)
        
    def generate_data(self, n, distribution, effect_size, k=3):
        groups = []
        for i in range(k):
            loc = i * effect_size 
            if distribution == 'normal': data = self.rng.normal(loc=loc, scale=1.0, size=n)
            elif distribution == 'skewed': data = self.rng.lognormal(mean=loc, sigma=0.5, size=n)
            elif distribution == 'heavy_tailed': data = self.rng.standard_t(df=3, size=n) + loc
            elif distribution == 'uniform': data = self.rng.uniform(low=loc-1.732, high=loc+1.732, size=n)
            elif distribution == 'bimodal':
                comp1 = self.rng.normal(loc=-1 + loc, scale=0.5, size=n//2)
                comp2 = self.rng.normal(loc=1 + loc, scale=0.5, size=n - n//2)
                data = np.concatenate([comp1, comp2])
            else: data = self.rng.normal(loc=loc, scale=1.0, size=n)
            groups.append(data.tolist())
        return groups

    def run(self, test_name, sample_sizes, indeterminacy_levels, distributions, effect_sizes, alpha=0.05, progress_callback=None):
        results = []
        total_conditions = len(sample_sizes) * len(indeterminacy_levels) * len(distributions) * len(effect_sizes)
        total_iters = total_conditions * self.n_simulations
        current_iter = 0
        
        for n in sample_sizes:
            for delta in indeterminacy_levels:
                for dist in distributions:
                    for effect in effect_sizes:
                        
                        orig_powers, mod_powers = [], []
                        orig_decisions, mod_decisions = [], []
                        orig_widths, mod_widths = [], []
                        
                        for sim in range(self.n_simulations):
                            self.rng = np.random.default_rng(seed=self.random_seed + current_iter)
                            current_iter += 1
                            
                            k = 2 if test_name == 'mann_whitney' else 3
                            raw_groups = self.generate_data(n, dist, effect, k=k)
                            neutro_groups = []
                            
                            for g in raw_groups:
                                n_arr = neutrosophicate(g, indeterminacy_threshold=0.0) 
                                if delta > 0:
                                    n_indet = int(n * delta)
                                    indices = self.rng.choice(len(n_arr.data), size=n_indet, replace=False)
                                    for idx in indices:
                                        t_val = n_arr.data[idx].T[0]
                                        n_arr.data[idx] = NeutrosophicNumber((t_val, t_val), (0.0, 1.0), (0.0, 0.0))
                                neutro_groups.append(n_arr)
                                
                            if test_name == 'kruskal_wallis':
                                orig = kruskal_wallis_original(neutro_groups)
                                mod = kruskal_wallis_modified(neutro_groups)
                            elif test_name == 'mann_whitney':
                                orig = mann_whitney_original(neutro_groups[0], neutro_groups[1])
                                mod = mann_whitney_modified(neutro_groups[0], neutro_groups[1])
                            else: 
                                orig = moods_median_original(neutro_groups)
                                mod = moods_median_modified(neutro_groups)
                                
                            piv_orig = orig.get('p_interval', orig.get('p_interval_original', (1,1)))
                            piv_mod = mod.get('p_interval', mod.get('p_interval_modified', (1,1)))
                            
                            orig_decisions.append(orig['decision_zone'])
                            mod_decisions.append(mod['decision_zone'])
                            orig_widths.append(piv_orig[1] - piv_orig[0])
                            mod_widths.append(piv_mod[1] - piv_mod[0])
                            
                            orig_powers.append(1 if orig['decision_zone'] == 'Reject H0' else 0)
                            mod_powers.append(1 if mod['decision_zone'] == 'Reject H0' else 0)
                            
                            if progress_callback and (current_iter % 50 == 0):
                                progress_callback(current_iter, total_iters)
                                
                        def calc_metrics(powers, decisions, widths, variant):
                            power = np.mean(powers) if effect > 0 else 0
                            type1 = np.mean(powers) if effect == 0 else 0
                            stability = 1 - (decisions.count('Indeterminate Decision') / max(1, len(decisions)))
                            mean_width = np.mean(widths)
                            return power, type1, stability, mean_width
                            
                        opower, otype1, ostability, owidth = calc_metrics(orig_powers, orig_decisions, orig_widths, 'original')
                        mpower, mtype1, mstability, mwidth = calc_metrics(mod_powers, mod_decisions, mod_widths, 'modified')
                        
                        relative_eff = mpower / max(opower, 0.001) if effect > 0 else 1.0
                        
                        results.append({
                            'test': test_name, 'variant': 'original', 'n': n, 'delta': delta,
                            'distribution': dist, 'effect_size': effect,
                            'power': opower, 'type1_error': otype1,
                            'decision_stability': ostability, 'interval_width': owidth,
                            'relative_efficiency': 1.0
                        })
                        results.append({
                            'test': test_name, 'variant': 'modified', 'n': n, 'delta': delta,
                            'distribution': dist, 'effect_size': effect,
                            'power': mpower, 'type1_error': mtype1,
                            'decision_stability': mstability, 'interval_width': mwidth,
                            'relative_efficiency': relative_eff
                        })
                        
        df = pd.DataFrame(results)
        if progress_callback: progress_callback(total_iters, total_iters)
        return df
