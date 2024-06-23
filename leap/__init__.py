# fmt: off
import leap.dataset as dataset
import leap.error as error
import leap.plot as plot
import leap.utils.commons as commons
from leap.environment import env


def _get_njobs(n_jobs):
    return env["N_JOBS"] if n_jobs is None else n_jobs

# fmt: on
