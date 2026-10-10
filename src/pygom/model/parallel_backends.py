from dask.distributed import Client
from concurrent.futures import ProcessPoolExecutor

_dask_client = None

def get_dask_client():
    global _dask_client
    if _dask_client is None:
        _dask_client = Client(processes=True)   # local cluster
    return _dask_client


# --- IMPORTANT: top-level, picklable helper ---
def _execute_task(func, pos_args, kw_args):
    return func(*pos_args, **kw_args)


def run_multiprocessing(func, args_list, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_execute_task, func, pos, kw)
            for (pos, kw) in args_list
        ]
        return [f.result() for f in futures]


def run_dask(func, args_list):
    client = get_dask_client()
    futures = [
        client.submit(func, *pos, **kw)
        for (pos, kw) in args_list
    ]
    return client.gather(futures)