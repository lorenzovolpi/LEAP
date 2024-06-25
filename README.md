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
python -m leap.experiments.run
```

To generate the plots run:

```bash
python -m leap.experiments.plotting
```

To generate the tables run:

```bash
python -m leap.experiments.tables
```

The results for the experiments, the plots and the tables will be saved, respectively, in `results`, `plots` and `tables`, all rooted in the current folder.  
To change the output root folder export the `PHD_OUT_DIR` environment variable before running the above commands.
