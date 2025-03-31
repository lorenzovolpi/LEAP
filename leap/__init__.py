# fmt: off
import leap.commons as commons
import leap.data as data
import leap.error as error
import leap.plot as plot
from leap.environment import env


def _get_njobs(n_jobs):
    return env["N_JOBS"] if n_jobs is None else n_jobs

# fmt: on
