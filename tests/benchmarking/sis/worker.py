# worker.py

from multiprocessing import Pool
import pandas as pd
from stochastic_simulation import run_simulation
import sys
import time
import os

start = time.perf_counter()

task_id = int(sys.argv[1])
runs_per_task = int(sys.argv[2])
N = int(sys.argv[3])
gamma = float(sys.argv[4])
R0 = float(sys.argv[5])
method = str(sys.argv[6])

beta = R0 * gamma

# SIS equilibrium
i0 = int(N * (1 - (1 / R0) ) )

params = {
    'N': N,
    'gamma': gamma,
    'beta': beta,
    'i0': i0,
    'method': method
}

R0_str = f"{R0:.3f}".rstrip('0').rstrip('.').replace('.', 'p')

if method == "fixed_tau":
    param = float(sys.argv[7])
    params['tau'] = param
    param_str = f"{param:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
    fn = f"{R0_str}_{method}_{param_str}"

elif method == "cao2006":
    param = float(sys.argv[7])
    params['epsilon'] = param
    param_str = f"{param:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
    fn = f"{R0_str}_{method}_{param_str}"

elif method == "ukhsa2026":
    param = float(sys.argv[7])
    params['epsilon'] = param
    param_str = f"{param:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
    fn = f"{R0_str}_{method}_{param_str}"

elif method == "direct":
    fn = f"{R0_str}_direct"

start_run = task_id * runs_per_task
run_ids = range(start_run, start_run + runs_per_task)

def wrapper(run_id):
    return run_simulation(
        run_id=run_id,
        seed=run_id,
        params=params
    )

ncpus = int(os.environ["SLURM_CPUS_PER_TASK"])

if __name__ == "__main__":

    with Pool(ncpus) as pool:
        results = pool.map(wrapper, run_ids)

    pd.DataFrame(results).to_csv(
        f"sis_results_{fn}_{task_id:05d}.csv",
        index=False
    )

elapsed = time.perf_counter() - start

print(f"Completed {runs_per_task} simulations "
      f"in {elapsed:.1f} seconds")
print(f"Rate = {runs_per_task/elapsed:.1f} sims/sec")