# Sandbox for LLM Prototyping for Autonomous Vehicles

## Setup

### Installation

Clone the repository and its submodules with

```
git clone --recurse-submodules https://github.com/cpsl-research/av-llm-sandbox.git
```

This repository requires use of the [`uv`][uv] tool to set up the python environment. Install it with

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

[`uv`][uv] will automatically set up the environment when you run a script, but to manually install the environment and to ensure appropriate setting of dependencies, you can run the following:

```
uv lock
uv sync
```

### Download datasets

If you do not already have the AV datasets downloaded on your machine, you can use our scripts. If you already have them downloaded, ensure that the paths in the notebooks are set appropriately. To download the datasets, run the following, replacing `path_to_datasets` with your desired path to the datasets (e.g., `/data/shared/` - without the dataset suffix)

```
cd data
./download_nuscenes.sh path_to_datasets mini
```

### Running noteboks in VSCode

VSCode can use the [`uv`][uv] environment to run notebooks interactively. Ensure that you install the environment first, then open up your notebook in VSCode. One of the environments listed should be `.venv/bin/python`. Select that one and you should have access to all the installed packages.


## Generate AV-LLM Dataset

To generate the AV-LLM dataset, run the following, replacing `path_to_datasets` with your path to the data (e.g., `/data/shared/nuScenes`)

```
cd scripts
uv run make_dataset.py path_to_datasets --output_prefix dataset --dataset nuscenes
```


## Best Practices

### Mitigating notebook version inflation

To mitigate the inflation of repository size with jupyter notebooks and their contents, we've included a pre-commit hook that clears all cell outputs. Ensure that you configure utilization of the hook before making any commits to the repo by running

```
git config core.hooksPath hooks
```

The pre-commit hook specifies the location of any jupyter notebooks in the hook and clears outputs. If you put a notebook in a non-standard location, ensure to update the information in the pre-commit hook.


[uv]: https://docs.astral.sh/uv