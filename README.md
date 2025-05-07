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

### Dataset description

The dataset is organized as a dictionary with the following entries. Double letter entries (e.g., `DD`) are placeholders. 

```
"dataset"
    "scene_AA"                      # str, dictionary key
        "agent_BB"                  # str, dictionary key
            "frame_CC"              # str, dictionary key
                "frame"             # int
                "timestamp"         # float
                "image_paths"       # str, dictionary key
                    "CAM_DD"        # str, dictionary key
                "meta_actions"      # str, dictionary key
                    "dt_EE"         # str, dictionary key
                        action      # list of 2* ints or null
                "waypoints_3d"      # str, dictionary key
                    "dt_EE"         # str, dictionary key
                        pts_3d      # list of 3 floats
                "waypoints_pixel"   # str, dictionary key
                    "dt_EE"         # str, dictionary key
                        pts_pix     # list of 2 floats
                "agent_state"       # str, dictionary key
                    "FF"            # str, dictionary key
                        "position"  # list of 3 floats
                        "velocity"  # list of 3 floats
                        "speed"     # norm of velocity
                        "attitude"  # list of 4 floats
                        "yaw"       # float
"metadata":
    "action_table"          # str, dictionary key
        "GG"                # str, dictionary key
            "HH"            # int
    "reverse_action_table"  # str, dictionary key
        "II"                # str, dictionary key
            "JJ"            # str
    "dataset"
        "KK"
    "waypoints_3d_reference"
        "LL"
```

Expanding notes:
```
dataset
------------
dt_EE       -->  look-ahead time in seconds (e.g., dt_2 is 2 second look-ahead)
FF          -->  reference frame, one of [global, local, diff]
action      -->  integer codifying the action (see lookup table)
pts_3d      -->  future position difference from current position in 3D
pts_pix     -->  future position difference from current position projected onto
                 2D image
attitude    -->  4 dimensional quaternion
yaw         -->  heading angle (radians)

metadata
------------
action_table
    GG                  -->  str name of the action
        HH              -->  int for the action 
reverse_action_table
    II                  -->  integer for action, **but json requires keys to be str
                             so actually a str**
        JJ              -->  str name of the action
dataset
    KK                  -->  name of the dataset
waypoints_3d_reference
    LL                  -->  reference frame for the waypoints. "camera" means a camera
                             reference frame with the convention: (X -> right, Y ->
                             down, Z -> forward) in the driving frame.
```

## Best Practices

### Mitigating notebook version inflation

To mitigate the inflation of repository size with jupyter notebooks and their contents, we've included a pre-commit hook that clears all cell outputs. Ensure that you configure utilization of the hook before making any commits to the repo by running

```
git config core.hooksPath hooks
```

The pre-commit hook specifies the location of any jupyter notebooks in the hook and clears outputs. If you put a notebook in a non-standard location, ensure to update the information in the pre-commit hook.


[uv]: https://docs.astral.sh/uv