# LEAP

## Setup

To setup the environment for running the experiments, first clone the repository:

```bash
git clone https://github.com/lorenzovolpi/LEAP.git leap
cd leap
```

then create the python virtual environment and install requirements:

```bash
python -m venv .venv
chmod +x .venv/bin/activate
source ./.venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

To run the experiments shown in the cited paper, run:

```bash
python -m exp.main
```

To generate the plots run:

```bash
python -m exp.plot    # for diagonal plots relating true and estimated accuracy
python -m exp.ctdiff  # for heatmap plots showint contingency table errors
python -m exp.times   # for inference time plots
```

To generate the tables run:

```bash
python -m exp.table
```

The `output` folder contains the subfolders where the results for the experiments 
(`data`), the diagonal plots (`plots`), the heatmaps (`ctdiff`), the inference time 
plots (`times`) and the tables (`tables`) will be stored.
To change the output root folder export the `LEAP_OUT_DIR` environment variable 
before running the above commands.

The execution of the tests is parallelised, with each combination of (`classifier`, `dataset`, `method`)
running in a separate process. The `LEAP_N_JOBS` environment variable controls the number
of availables processors to dedicate to the execution.
