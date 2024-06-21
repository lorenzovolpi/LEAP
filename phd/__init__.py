# fmt: off
import phd.dataset as dataset
import phd.error as error
import phd.plot as plot
import phd.utils.commons as commons
from phd.environment import env


def _get_njobs(n_jobs):
    return env["N_JOBS"] if n_jobs is None else n_jobs

# fmt: on
