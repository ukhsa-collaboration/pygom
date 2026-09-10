# worker.py

import pandas as pd
from stochastic_simulation import run_simulation
import sys

task_id = int(sys.argv[1])
runs_per_task = int(sys.argv[2])
N = int(sys.argv[3])
gamma = float(sys.argv[4])
R0 = float(sys.argv[5])
tau = float(sys.argv[6])
i0 = int(sys.argv[7])

start_run = task_id * runs_per_task

beta = R0 * gamma
params = {
    'N': N,
    'gamma': gamma,
    'beta': beta,
    'tau': tau,
    'i0': i0
}

results = []

for i in range(runs_per_task):

    run_id = start_run + i

    results.append(
        run_simulation(
            run_id=run_id,
            seed=run_id,
            params=params
        )
    )

df = pd.DataFrame(results)

df.to_csv(
    f"results_{task_id:05d}.csv",
    index=False
)