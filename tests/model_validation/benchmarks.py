import numpy as np

class GrowthRateBenchmark:
    def __init__(self):
        self.name = "growth_rate"

    def evaluate(self, scenario, solutions):
        observed_r = []
        for solution in solutions:
            # If stochastically extinct before exponential growth, skip
            rec = scenario.cumulative_infected_population(solution)
            if rec[-1] < 100:
                continue

            inf = scenario.infectious_population(solution)
            t = solution.result.t

            i_peak = np.argmax(inf)
            inf_peak = inf[i_peak]
            t_peak = t[i_peak]
            
            Imin = max(50, inf_peak * 0.01)
            Imax = max(100, inf_peak * 0.1)

            # First times these thresholds are crossed:
            i1 = np.where(inf >= Imin)[0][0]
            i2 = np.where(inf >= Imax)[0][0]

            mask = np.zeros_like(inf, dtype=bool)
            mask[i1:i2+1] = True

            if np.sum(mask) < 5:
                continue

            coef = np.polyfit(
                t[mask],
                np.log(inf[mask]),
                deg=1
            )
            r = coef[0]

            observed_r.append(r)

        assert len(observed_r) > 0

        mean_observed_r = np.mean(observed_r)

        expected_r = scenario.theoretical_growth_rate()

        assert (
            abs(mean_observed_r - expected_r) / expected_r
            < scenario.growth_rate_tol
        ), (
            f"Expected growth rate {expected_r:.4f}, "
            f"observed {mean_observed_r:.4f}"
        )

class FinalSizeBenchmark:
    def __init__(self):
        self.name = "final_size"

    def evaluate(self, scenario, solutions):
        observed_size = []
        for solution in solutions:
            inf_cml = scenario.cumulative_infected_population(solution)
            observed_size.append(inf_cml[-1])

        assert len(observed_size) > 0

        mean_observed_final_size = np.mean(observed_size)

        expected_final_size = scenario.theoretical_final_size_mn()

        assert (
            abs(mean_observed_final_size - expected_final_size) / expected_final_size
            < scenario.final_size_tol
        ), (
            f"Expected final size {expected_final_size:.4f}, "
            f"observed {mean_observed_final_size:.4f}"
        )


class EndemicMeanBenchmark:
    def __init__(self):
        self.name = "endemic_mean"

    def evaluate(self, scenario, solutions):
        observed_endemic_mn = []
        for solution in solutions:
            inf = scenario.infectious_population(solution)
            n = len(inf)

            # Discard first half as burn in
            i = n // 2
            inf = inf[i:]

            observed_endemic_mn.append(inf.mean())

        assert len(observed_endemic_mn) > 0

        mean_observed_endemic_mn = np.mean(observed_endemic_mn)

        expected_endemic_mn = scenario.theoretical_endemic_mean()

        assert (
            abs(mean_observed_endemic_mn - expected_endemic_mn) / expected_endemic_mn
            < scenario.endemic_mn_tol
        ), (
            f"Expected final size {expected_endemic_mn:.4f}, "
            f"observed {mean_observed_endemic_mn:.4f}"
        )

class EndemicSDBenchmark:
    def __init__(self):
        self.name = "endemic_sd"

    def evaluate(self, scenario, solutions):
        observed_endemic_sd = []
        for solution in solutions:
            inf = scenario.infectious_population(solution)
            n = len(inf)

            # Discard first half as burn in
            i = n // 2
            inf = inf[i:]

            observed_endemic_sd.append(inf.std())

        assert len(observed_endemic_sd) > 0

        mean_observed_endemic_sd = np.mean(observed_endemic_sd)

        expected_endemic_sd = scenario.theoretical_endemic_sd()

        assert (
            abs(mean_observed_endemic_sd - expected_endemic_sd) / expected_endemic_sd
            < scenario.endemic_sd_tol
        ), (
            f"Expected final size {expected_endemic_sd:.4f}, "
            f"observed {mean_observed_endemic_sd:.4f}"
        )