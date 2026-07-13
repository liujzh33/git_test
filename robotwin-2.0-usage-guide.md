# RoboTwin 2.0 Usage Guide (Full Text)

> Aggregated from [RoboTwin 2.0 Usage Guide](https://robotwin-platform.github.io/doc/usage/index.html) and all Usage subpages. Images keep remote URLs.


---

## Page: Usage Guide

> Source: [https://robotwin-platform.github.io/doc/usage/index.html](https://robotwin-platform.github.io/doc/usage/index.html)

# RoboTwin 2.0 Usage Guide

This documentation provides a comprehensive guide to using RoboTwin 2.0, covering environment setup, data collection and configuration, policy deployment, usage of demo policies, automatic code generation for new tasks, API tutorial, language instruction generation, and digital asset annotation.


---

## Page: Install & Download

> Source: [https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

# Install & Download

## 1. **Dependencies**

System Support:

We currently best support Linux based systems. There is limited support for windows and no support for MacOS at the moment. We are working on trying to support more features on other systems but this may take some time. Most constraints stem from what the [SAPIEN](https://github.com/haosulab/SAPIEN/) package is capable of supporting.

| System / GPU | CPU Sim | GPU Sim | Rendering |
| --- | --- | --- | --- |
| Linux / NVIDIA GPU | ✅ | ✅ | ✅ |
| Windows / NVIDIA GPU | ✅ | ❌ | ✅ |
| Windows / AMD GPU | ✅ | ❌ | ✅ |
| WSL / Anything | ✅ | ❌ | ❌ |
| MacOS / Anything | ✅ | ❌ | ✅ |

> Occasionally, data collection may get stuck when using A/H series GPUs. This issue may be related to [RoboTwin issue #83](https://github.com/RoboTwin-Platform/RoboTwin/issues/83#issuecomment-3012135745) and [SAPIEN issue #219](https://github.com/haosulab/SAPIEN/issues/219).

Python versions:

- Python 3.10

CUDA version:

- 12.1 (Recommended)

Hardware:

- Rendering: NVIDIA or AMD GPU
- Ray tracing: NVIDIA RTX GPU or AMD equivalent
- Ray-tracing Denoising: NVIDIA GPU
- GPU Simulation: NVIDIA GPU

Software:

- Ray tracing: NVIDIA Driver >= 470
- Denoising (OIDN): NVIDIA Driver >= 520

### 1.1 Additional Requirements for Docker

When running in a Docker container, ensure that the following environment variable is set when starting the container:

```
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

Important : The graphics capability is essential. Omitting it may result in segmentation faults due to missing Vulkan support.

For more information, see [HERE](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

## 2. Install Vulkan (if not installed)

```
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

Check by running `vulkaninfo`

## 3. Basic Env

First, prepare a conda environment.

```
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```

RoboTwin 2.0 Code Repo: <https://github.com/RoboTwin-Platform/RoboTwin>

```
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
```

Then, run `script/_install.sh` to install basic envs and CuRobo:

```
bash script/_install.sh
```

If you meet curobo config path issue, try to run `python script/update_embodiment_config_path.py`

If you encounter any problems, please refer to the [manual installation](#manual-installation-only-when-step-2-failed) section. If you are not using 3D data, a failed installation of pytorch3d will not affect the functionality of the project.

If you haven't installed ffmpeg, please turn to <https://ffmpeg.org/>. Check it by running `ffmpeg -version`.

## 4. Download Assets (RoboTwin-OD, Texture Library and Embodiments)

To download the assets, run the following command. If you encounter any rate-limit issues, please log in to your Hugging Face account by running `huggingface-cli login`:

```
bash script/_download_assets.sh
```

The structure of the `assets` folder should be like this:

```
assets
├── background_texture
├── embodiments
│   ├── embodiment_1
│   │   ├── config.yml
│   │   └── ...
│   └── ...
├── objects
└── ...
```

## 5. Manual Installation (Only when step 3 failed)

1. Install requirements

   ```
   pip install -r requirements.txt
   ```
2. Install pytorch3d

   ```
   pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
   ```
3. Install CuRobo

   ```
   cd envs
   git clone https://github.com/NVlabs/curobo.git
   cd curobo
   pip install -e . --no-build-isolation
   cd ../..
   ```
4. Adjust code in `mplib` (**Important**)
5. You can use `pip show mplib` to find where the `mplib` installed.
6. Remove `or collide`

```
# mplib.planner (mplib/planner.py) line 807
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```


---

## Page: Collect Data

> Source: [https://robotwin-platform.github.io/doc/usage/collect-data.html](https://robotwin-platform.github.io/doc/usage/collect-data.html)

# Collect Data

We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

Before collecting data, please check the common issue . We strongly recommand you to avoid using A/H/V series GPUs to collect and evaluate your policy.

Before collecting data, please review common issue #3, [Stuck While Collecting Data and Evaluating](https://robotwin-platform.github.io/doc/common-issue/index.html). We strongly recommend avoiding A-, H-, or V-series GPUs for data collection and policy evaluation.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Clean Data Example: bash collect_data.sh beat_block_hammer demo_clean 0
# Radomized Data Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

After data collection is completed, the collected data will be stored under `data/${task_name}/${task_config}`.

**An episode's data will be stored in one HDF5 file. Specifically, the images will be stored as bit streams. If you want to recover the image, you can use the following code:**

```
image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
```

- Each trajectory's observation and action data are saved in **HDF5 format** in the `data` directory.
- The corresponding **language instructions** for each trajectory are stored in the `instructions` directory.
- **Head camera videos** of each trajectory can be found in the `video` directory.
- The `_traj_data`, `.cache`, `scene_info.json`, and `seed.txt` files are auxiliary outputs generated during the data collection process.

All available `task_name` options can be found in the [documentation](https://robotwin-platform.github.io/doc/tasks/index.html). The `gpu_id` parameter specifies which GPU to use and should be set to an integer in the range `0` to `N-1`, where `N` is the number of GPUs available on your system.

Our data synthesizer enables automated data collection by executing the task scripts in the `envs` directory, in combination with the `curobo` robot planner. Specifically, data collection is configured through a task-specific configuration file (see the tutorial in `./configurations.md`), which defines parameters such as the target embodiment, domain randomization settings, and the number of data samples to collect.

The success rate of data generation for each embodiment across all tasks can be found at: <https://robotwin-platform.github.io/doc/tasks/index.html>. Due to the structural limitations of different robotic arms, not all embodiments are capable of completing every task.

Our pipeline first explores a set of random seeds (`seed.txt`) to identify trajectories that can yield successful data collection. It then records fine-grained action trajectories (`_traj_data`) accordingly. Collected videos are available in the `videos` directory.

The entire process is fully automated—just run a single command to get started.

> ⚠️ The `missing pytorch3d` warning can be ignored if 3D data is not required.


---

## Page: Domain Randomization

> Source: [https://robotwin-platform.github.io/doc/usage/domain-randomization.html](https://robotwin-platform.github.io/doc/usage/domain-randomization.html)

# Domain Randomization

RoboTwin’s domain randomization primarily focuses on scene clutter, random lighting, over 12,000 tabletop textures, randomized tabletop heights, and camera viewpoint perturbations. The corresponding configuration options can be found at: 👉 [RoboTwin 2.0 Document (Usage: Configurations)](https://robotwin-platform.github.io/doc/usage/configurations.html)

![description](https://robotwin-platform.github.io/doc/usage/./images/domain_randomization.png)


---

## Page: Configurations

> Source: [https://robotwin-platform.github.io/doc/usage/configurations.html](https://robotwin-platform.github.io/doc/usage/configurations.html)

# Configuration Tutorial

All configuration files are stored in the `task_config` folder and follow the standard YAML format.

You can run `bash task_config/create_task_config.sh ${task_config_name}` to create new task configuration.

## 1. ✅ Minimal Example

**An episode's data will be stored in one HDF5 file. Specifically, the images will be stored as bit streams. If you want to recover the image, you can use the following code:**

```
image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
```

Below is a minimal configuration file to start a typical data collection session:

```
render_freq: 0
episode_num: 50
use_seed: false
save_freq: 15
embodiment:
- aloha-agilex
language_num: 100
domain_randomization:
  random_background: true
  cluttered_table: true
  clean_background_rate: 0.02
  random_head_camera_dis: 0
  random_table_height: 0.03
  random_light: true
  crazy_random_light_rate: 0.02
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./data
clear_cache_freq: 5
collect_data: true
eval_video_log: true
```

---

## 2. 🔧 Configuration Breakdown

### 2.1 🎯 Task & Embodiment Settings

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `embodiment` | list | ✅ | List of robot embodiment(s). For a dual-arm robot, use `[name]`, e.g., `[aloha-agilex]`; to combine two single-arm robots, use `[left, right, interval]`, e.g., `embodiment: [piper, franka-panda, 0.6]`, `embodiment: [franka-panda, franka-panda, 0.8]`. The `interval` specifies the distance between arms (typically 0.6–0.8 meters). Available Embodiment: `ur5-wsg`, `ARX-X5`, `franka-panda`, `piper`, `aloha-agilex`(dual-arm) |
| `use_seed` | bool | ✅ | Whether to use a predefined seed list from `data/${task_name}/${task_config}/seed.txt`. If `false`, the system will automatically explore viable seeds. |
| `episode_num` | int | ✅ | Number of **successful episodes** to collect. |
| `language_num` | int | optional | If using language-conditioned task planning, sets the number of language descriptions to sample for each task. |

### 2.2 🧠 Domain Randomization

Configure task variation for better generalization.

```
domain_randomization:
  random_background: true
  cluttered_table: true
  clean_background_rate: 0.02
  random_head_camera_dis: 0
  random_table_height: 0.03
  random_light: true
  crazy_random_light_rate: 0.02
  random_embodiment: false
```

| Field | Type | Description |
| --- | --- | --- |
| `random_background` | bool | Enable random textures for the table and background. |
| `cluttered_table` | bool | Add distractor objects to the table to simulate a cluttered environment. |
| `clean_background_rate` | float | Ratio of clean backgrounds (e.g., `0.02` = 2%). Only effective if `random_background` is `true`. |
| `random_head_camera_dis` | float | Random displacement applied to the head camera position (in meters). |
| `random_table_height` | float | Random variation in the table height (in meters). |
| `random_light` | bool | Enable randomized lighting during simulation. |
| `crazy_random_light_rate` | float | Probability of applying extreme lighting. Only effective if `random_light` is `true`. |
| `random_embodiment` | bool | Enable embodiment randomization (experimental, currently not fully supported). |

### 2.3 📷 Camera Configuration

```
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
```

| Field | Type | Description |
| --- | --- | --- |
| `head_camera_type` | str | Camera used for global observation. Options: see `task_config/_camera_config.yml`. |
| `wrist_camera_type` | str | Camera used for close-up view. |
| `collect_head_camera` | bool | Whether to collect head-view data. |
| `collect_wrist_camera` | bool | Whether to collect wrist-view data. |

---

### 2.4 📦 Data Collection Settings

| Field | Type | Description |
| --- | --- | --- |
| `collect_data` | bool | Enable actual data saving. |
| `save_freq` | int | Save every N steps. Per-step indicates 0.004s in the real world. |
| `save_path` | str | Directory to save data. Default: `./data`. |
| `clear_cache_freq` | int | Controls the frequency (in episodes) at which the Sapien scene cache is cleared. This helps manage GPU memory usage, especially when domain randomization is enabled and many diverse assets accumulate in memory. A smaller value (e.g., 1) increases clearing frequency but incurs additional time cost. |
| `eval_video_log` | bool | Save evaluation videos for replay. |

---

### 2.5 💾 Data Type

Specify which data to collect in each episode:

```
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: false
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
```

| Type | Description |
| --- | --- |
| `rgb` | RGB image from multiple views. |
| `third_view` | Third-person video. |
| `depth` | Depth images from cameras (mm). |
| `pointcloud` | Merged point cloud of the scene. |
| `observer` | Observer-view RGB frame. |
| `endpose` | end-effector pose in the world coordinate frame and gripper opening ratio. |
| `qpos` | Robot joint angles. |
| `mesh_segmentation` | Per-object segmentation from mesh. |
| `actor_segmentation` | Per-actor segmentation from RGB. |

##### 2.5.1 Note

- `endpose` will get an dict containing `left_endpose`, `left_gripper`, `right_endpose` and `right_gripper`. The `left_endpose` and `right_endpose` are list of 7 elements represent the position in world and orientation of the end-effector, following the order `x, y, z, qw, qx, qy, qz`. And the `left_gripper` and `right_gripper` are float numbers, which repersent the opening ratio of the gripper, ranging from 0 to 1. The rotation of end-effector is as the image below: for all embodiments, the end-effector rotation is consistent, with the x-axis pointing across the gripper and the z-axis pointing across the camera.

## End-Effector Rotation

### 2.6 🔍 Point Cloud Settings

| Field | Type | Description |
| --- | --- | --- |
| `pcd_down_sample_num` | int | FPS (Farthest Point Sampling) number; set `0` to keep all points. |
| `pcd_crop` | bool | Whether to crop out table/walls based on known transforms. |

---

### 2.7 🎥 Rendering

| Field | Type | Description |
| --- | --- | --- |
| `render_freq` | int | Render visualization every N steps. Set to `0` to disable. For servers without display, recommend `0`. If you want to visualize the task, try to modify it to `20` (as example) |

---

## 3. 📌 Notes

- All task names must correspond to files in `env/<task_name>.py`.
- For available embodiments and cameras, refer to:
- `task_config/_embodiment_config.yml`
- `task_config/_camera_config.yml`
- The system supports both dual-arm and single-arm setups.
- Seeds, if used, are located in `task_config/seeds/`.


---

## Page: Control Robot

> Source: [https://robotwin-platform.github.io/doc/usage/control-robot.html](https://robotwin-platform.github.io/doc/usage/control-robot.html)

# Control Robot

The `take_action` function in `_base_task` is used to control actions during task execution. It accepts two parameters: `action` and `action_type`.

## 1. Supported Action Types

The parameter `action_type` supports two modes:

- `qpos` (Joint Position Control) — **default**
- `ee` (End-Effector Pose Control)

Depending on the selected mode, the format and dimension of the input `action` will differ.

---

### 1.1 `qpos` Mode (Joint Position Control)

In `qpos` mode, the `action` is defined as:

```
[left_arm_joints + left_gripper + right_arm_joints + right_gripper]
```

- The specific dimension of the `action` depends on the robotic arm configuration.
- The system will **automatically adjust** the input dimensions during deployment to match the specific robot configuration.

---

### 1.2 `ee` Mode (End-Effector Pose Control)

In `ee` mode, the `action` is defined as:

```
[left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
```

- The dimension is **fixed**, regardless of the robot configuration.

---

## 2. Deployment Example

You can find a demonstration of usage in:

```
policy/Your_Policy/deploy_policy.py
```

This file provides a sample implementation to help you understand how to use the `take_action` function with different `action_type` settings during deployment.


---

## Page: Deploy Your Policy

> Source: [https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

# 🚀 Deploy Your Policy

To deploy and evaluate your policy, you need to **modify the following three files**:

- `eval.sh`: [eval.sh demo](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/Your_Policy/eval.sh)
- `deploy_policy.yml`: [deploy\_policy.yml demo](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/Your_Policy/deploy_policy.yml)
- `deploy_policy.py`: [deploy\_policy.py demo](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/Your_Policy/deploy_policy.py)

In `deploy_policy.py`, the following components are defined: `get_model` for loading the policy model, `encode_obs` for observation processing (modification may not be necessary), and `get_action` along with the control loop that handles observation acquisition and action execution.

The `deploy_policy.yml` file specifies the input parameters, which are eventually passed into the `get_model` function as `usr_args` to assist in locating, defining, and loading your model.

In `eval.sh`, the parameters specified after `overrides` can be used to overwrite those in `deploy_policy.yml`, allowing you to specify different settings without manually modifying the YAML file each time.

```
# policy/Your_Policy/deploy_policy.py
# import packages and module here

def encode_obs(observation):  # Post-Process Observation
    obs = observation
    # ...
    return obs

def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    Your_Model = None
    # ...
    return Your_Model  # return your policy model

def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    if len(
            model.obs_cache
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified

def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
```

---

## 1. 🔧 `deploy_policy.yml`

You are free to **add any parameters** needed in `deploy_policy.yml` to specify your model setup (e.g., checkpoint path, model type, architecture details). The entire YAML content will be passed to `deploy_policy.py` as `usr_args`, which will be available in the `get_model()` function.

---

## 2. 🖥️ `eval.sh`

Update the script to pass additional arguments to override default values in `deploy_policy.yml`.

```
#!/bin/bash

policy_name=Your_Policy
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
# [TODO] Add your custom command-line arguments here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to project root

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} 
    # [TODO] Add your custom arguments here
```

---

## 3. 🧠 `deploy_policy.py`

You need to implement the following methods in `deploy_policy.py`:

### 3.1 `encode_obs(obs: dict) -> dict`

Optional. This function is used to preprocess the raw environment observation (e.g., color channel normalization, reshaping, etc.). If not needed, it can be left unchanged.

---

### 3.2 `get_model(usr_args: dict) -> Any`

Required. This function receives the full configuration from `deploy_policy.yml` via `usr_args` and must return the initialized model. You can define your own loading logic here, including parsing checkpoints and network parameters.

---

### 3.3 `eval(env, model, observation, instruction) -> Any`

Required. The main evaluation loop. Given the current environment instance, model, and observation (as a dictionary), and a natural language `instruction` (string), this function must compute the next action and execute it in the environment.

---

### 3.4 `update_obs(obs: dict) -> None`

Optional. Used to update any internal state of the model or observation buffer. Useful if your model requires a history of frames or a memory-based context.

---

### 3.5 `get_action(model, obs: dict) -> Any`

Optional. Given a model and current observation, return the action to be executed. This is useful if action computation is separated from the evaluation loop.

---

### 3.6 `reset_model() -> None`

Optional but **recommended**. This function is called before the evaluation of **each episode**, allowing you to reset model states such as recurrent memory, history buffers, or context encodings.

---

## 4. ✔️ Run `eval.sh`

```
bash eval.sh ...(input parameters you define)
```

## 5. 📌 Notes

- The variable `instruction` is a string containing the language command describing the task. You can choose how (or whether) to use it.
- Your policy should be compatible with the input/output format expected by the simulator.


---

## Page: ACT

> Source: [https://robotwin-platform.github.io/doc/usage/ACT.html](https://robotwin-platform.github.io/doc/usage/ACT.html)

# ACT (Action Chunking Transformer)

## 1. Install

```
cd policy/ACT

pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython

cd detr && pip install -e . && cd ..
```

## 2. Prepare Training Data

This step performs data preprocessing, converting the original **RoboTwin 2.0** data into the format required for ACT training. The `expert_data_num` parameter specifies the number of trajectory pairs to be used as training data.

```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_clean 50
```

## 3. Train Policy

This step launches the training process. By default, the model is trained for **6,000 steps**.

```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# bash train.sh beat_block_hammer demo_clean 50 0 0
```

## 4. Eval Policy

The `task_config` field refers to the **evaluation environment configuration**, while the `ckpt_setting` field refers to the **training data configuration** used during policy learning.

```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: DP

> Source: [https://robotwin-platform.github.io/doc/usage/DP.html](https://robotwin-platform.github.io/doc/usage/DP.html)

# DP (Diffusion Policy)

## 1. Install

```
cd policy/DP
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor sympy
pip install -e .
```

## 2. Prepare Training Data

This step performs data preprocessing, converting the original **RoboTwin 2.0** data into the **Zarr format** required for DP training. The `expert_data_num` parameter specifies the number of trajectory pairs to be used as training data.

```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_clean 50
# or processing randomized data: bash process_data.sh beat_block_hammer demo_randomized 50
```

## 3. Train Policy

This step launches the training process. By default, the model is trained for **600 steps**. The `action_dim` parameter defines the dimensionality of the robot’s action space — for example, it is **14** for the `aloha-agilex` embodiment.

```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
# bash train.sh beat_block_hammer demo_clean 50 0 14 0
# For `aloha-agilex` embodiment, the action_dim is 14
```

## 4. Eval Policy

The `task_config` field refers to the **evaluation environment configuration**, while the `ckpt_setting` field refers to the **training data configuration** used during policy learning.

```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: DP3

> Source: [https://robotwin-platform.github.io/doc/usage/DP3.html](https://robotwin-platform.github.io/doc/usage/DP3.html)

# DP3 (3D Diffusion Policy)

> Since **DP3** is a 3D policy that requires point cloud input, please make sure to set `data_type/pointcloud` to `true` during data collection.

## 1. Install

```
cd policy/DP3/3D-Diffusion-Policy && pip install -e . && cd ..
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
```

## 2. Prepare Training Data

> If you meet `ZeroDivisionError: division by zero`: Since **DP3** is a 3D policy that requires point cloud input, please make sure to set `data_type/pointcloud` to `true` during data collection.

This step performs data preprocessing, converting the original **RoboTwin 2.0** data into the **Zarr format** required for DP3 training. The `expert_data_num` parameter specifies the number of trajectory pairs to be used as training data.

```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_clean 50
# or processing randomized data: bash process_data.sh beat_block_hammer demo_randomized 50
```

## 3. Train Policy

This step launches the training process. By default, the model is trained for **3,000 steps**.

```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# bash train.sh beat_block_hammer demo_clean 50 0 0
```

## 4. Eval Policy

The `task_config` field refers to the **evaluation environment configuration**, while the `ckpt_setting` field refers to the **training data configuration** used during policy learning.

```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: RDT

> Source: [https://robotwin-platform.github.io/doc/usage/RDT.html](https://robotwin-platform.github.io/doc/usage/RDT.html)

# RDT

## 1. Environment Setup

The conda environment for RDT with RoboTwin is identical to the official RDT environment. Please follow the ([RDT official documentation](https://github.com/thu-ml/RoboticsDiffusionTransformer)) to install the environment and directly overwrite the RoboTwin virtual environment in [INSTALLATION.md](https://robotwin-platform.github.io/doc/usage/../../INSTALLATION.md).

```
# Make sure python version == 3.10
conda activate RoboTwin

# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121

# Install packaging
pip install packaging==24.0
pip install ninja
# Verify Ninja --> should return exit code "0"
ninja --version; echo $?
# Install flash-attn
pip install flash-attn==2.7.2.post1 --no-build-isolation

# Install other prequisites
pip install -r requirements.txt
# If you are using a PyPI mirror, you may encounter issues when downloading tfds-nightly and tensorflow. 
# Please use the official source to download these packages.
# pip install tfds-nightly==4.9.4.dev202402070044 -i  https://pypi.org/simple
# pip install tensorflow==2.15.0.post1 -i  https://pypi.org/simple
```

## 2. Download Model

```
# In the ROOT directory
cd policy 
mkdir weights
cd weights
mkdir RDT && cd RDT
# Download the models used by RDT
huggingface-cli download google/t5-v1_1-xxl --local-dir t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-1b --local-dir rdt-1b
```

## 3. Collect RoboTwin Data

See [RoboTwin Tutorial (Usage Section)](https://robotwin-platform.github.io/doc/usage/collect-data.html) for more details.

## 4. Generate HDF5 Data

> HDF5 is the data format required for RDT training.

First, create the `processed_data` and `training_data` folders in the `policy/RDT` directory:

```
mkdir processed_data && mkdir training_data
```

Then, run the following in the `RDT/` root directory:

```
bash process_data_rdt.sh ${task_name} ${task_config} ${expert_data_num} ${gpu_id}
```

If success, you will find the `${task_name}-${task_config}-${expert_data_num}` folder under `policy/RDT/processed_data`.

## 5. Generate Configuration File

A `$model_name` manages the training of a model, including the training data and training configuration.

```
cd policy/RDT
bash generate.sh ${model_name}
# bash generate.sh RDT_demo_clean
```

This will create a folder named `\${model_name}` under training\_data and a configuration file `\${model_name}.yml` under model\_config.

### 5.1 Prepare Data

Copy all the data you wish to use for training from `processed_data` into `training_data/${model_name}`. If you have multiple tasks with different data, simply copy them in the same way.

Example folder structure:

```
training_data/${model_name}
├── ${task_1}
│   ├── episode_0
|   |   |── episode_0.hdf5
|   |   |-- instructions
|   │   │   ├── lang_embed_0.pt
|   │   │   ├── ...
├── ${task_2}
│   ├── ...
├── ...
```

### 5.2 Modify Training Config

In `model_config/${model_name}.yml`, you need to manually set the GPU to be used (modify `cuda_visible_device`). For a single GPU, try format like `0` to set GPU 0. For multi-GPU usage, try format like `0,1,4`. You can flexibly modify other parameters.

## 6. Finetune model

Once the training parameters are set, you can start training with:

```
bash finetune.sh ${model_name}
# bash finetune.sh RDT_demo_clean
```

**Note!**

If you fine-tune the model using a single GPU, DeepSpeed will not save `pytorch_model/mp_rank_00_model_states.pt`. If you wish to continue training based on the results of a single-GPU trained model, please set `pretrained_model_name_or_path` to something like `./checkpoints/${model_name}/checkpoint-${ckpt_id}`.

This will use the pretrain pipeline to import the model, which is the same import structure as the default `../weights/RDT/rdt-1b`.

## 7. Eval on RoboTwin

The `task_config` field refers to the **evaluation environment configuration**, while the `model_name` field refers to the **training data configuration** used during policy learning.

```
bash eval.sh ${task_name} ${task_config} ${model_name} ${checkpoint_id} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean RDT_demo_clean 10000 0 0
# This command trains the policy using the `RDT_demo_clean` setting ($model_name)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized RDT_demo_clean 10000 0 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: Pi0

> Source: [https://robotwin-platform.github.io/doc/usage/Pi0.html](https://robotwin-platform.github.io/doc/usage/Pi0.html)

# OpenPI

## 1. Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies,you can add uv your conda environment.

```
conda activate RoboTwin
# Install uv
pip install uv
```

Once uv is installed, run the following commands to set up the environment:

```
cd policy/pi0
# Install prequisites in uv environment
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

If you want to eval pi0 policy in RoboTwin，you are required to install curobo in your uv environment：

```
conda deactivate
source .venv/bin/activate
# At this point, you should be in the (openpi) environment
cd ../../envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../../policy/pi0/
bash
```

## 2. Generate RoboTwin Data

See [RoboTwin Tutorial (Usage Section)](https://robotwin-platform.github.io/doc/usage/collect-data.html) for more details.

## 3. Generate openpi Data

First, create the `processed_data` and `training_data` folders in the `policy/pi0` directory:

```
mkdir processed_data && mkdir training_data
```

Then, convert RoboTwin data to HDF5 data type.

```
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data_pi0.sh beat_block_hammer demo_clean 50
# or processing randomized data: bash process_data.sh beat_block_hammer demo_randomized 50
```

If success, you will find the `${task_name}-${task_config}-${expert_data_num}` folder under `policy/pi0/processed_data`.

Example folder structure:

```
processed_data/ 
├──${task_name}-${task_config}-${expert_data_num}
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...
```

Copy all the data you wish to use for training from `processed_data` into `training_data/${model_name}`. If you have multiple tasks with different data, simply copy them in the same way.please Place the corresponding task folders according to the example below.

```
#multi-task dataset example
training_data/  
├── ${model_name}
|       ├──${task_0}
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...
|       ├── ${task_1}
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...

#sigle task example
training_data/  
├── demo_clean
|       ├──beat_block_hammer-demo_clean-50
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...
```

Before generating the LerobotDataset format data for pi0,please make sure you have enough disk space under the `~/.cache`.This is because generating the `lerobotdataset` will require a large amount of space.And the datasets will be writed into `$XDG_CACHE_HOME`,which default path is `~/.cache`.If you don't have enough disk space under the `~/.cache` path, please use the following command to set a different cache directory with sufficient space:

```
export XDG_CACHE_HOME=/path/to/your/cache
```

Now, we can directly generate the LerobotDataset format data for pi0

```
# hdf5_path: The path to the generated HDF5 data (e.g., ./training_data/${model_name}/)
# repo_id: The name of the dataset (e.g., my_repo)
bash generate.sh ${hdf5_path} ${repo_id}
#bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

LerobotDataset format data will be writed into `${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}`

## 4. Write the Corresponding `train_config`

> For our official experiment, we use `pi0_base_aloha_robotwin_lora`

In `src/openpi/training/config.py`, there is a dictionary called `_CONFIGS`. You can modify 4 pre-configured PI0 configurations I’ve written: `pi0_base_aloha_robotwin_lora` `pi0_fast_aloha_robotwin_lora` `pi0_base_aloha_robotwin_full` `pi0_fast_aloha_robotwin_full`

You only need to write `repo_id` on your datasets.(e.g., `repo_id=demo_clean_repo`) If you want to change the `name` in `TrainConfig`, please include `fast` if you choose `pi_fast_base` model. If your do not have enough gpu memory, you can set `fsdp_devices`, refer to `config.py` line `src/openpi/training/config.py` line 352.

## 5. 5. Finetune model

```
# compute norm_stat for dataset
uv run scripts/compute_norm_stats.py --config-name ${train_config_name}
# uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full

# train_config_name: The name corresponding to the config in _CONFIGS, such as pi0_base_aloha_robotwin_full
# model_name: You can choose any name for your model
# gpu_use: if not using multi gpu,set to gpu_id like 0;else set like 0,1,2,3
bash finetune.sh ${train_config_name} ${model_name} ${gpu_use}
#bash finetune.sh pi0_base_aloha_robotwin_full demo_clean 0,1,2,3
```

| Training mode | Memory Required | Example GPU |
| --- | --- | --- |
| Fine-Tuning (LoRA) | > 46 GB | A6000(48G) |
| Fine-Tuning (Full) | > 100 GB | 2\*A100 (80GB) / 2\*H100 |

If your GPU memory is insufficient, please set the `fsdp_devices` parameter according to the following GPU memory reference, or reduce the `batch_size` parameter. Or you can try setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` in `finetune.sh`, it will cost lower gpu memory, but make training speed slower.

The default `batch_size` is 32 in the table below.

| GPU memory | Model type | GPU num | fsdp\_devices | Example GPU |
| --- | --- | --- | --- | --- |
| 24G | lora | 2 | 2 | 4090(24G) |
| 40G | lora | 2 | 2 | A100(40G) |
| 48G | lora | 1 | 1 | A6000(48G) |
| 40G | full | 4 | 4 | A100(40G) |
| 80G | full | 2 | 2 | A100(80G) |

## 6. Eval on RoboTwin

Checkpoints will be saved in policy/pi0/checkpoints/\({train\_config\_name}/\)}/${checkpoint\_id

You can modify the `deploy_policy.yml` file to change the `checkpoint_id` you want to evaluate.

```
# ckpt_path like: policy/pi0/checkpoints/pi0_base_aloha_robotwin_full/demo_clean/30000
bash eval.sh ${task_name} ${task_config} ${train_config_name} ${model_name} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_full demo_clean 0 0
# This command trains the policy using the `demo_clean` setting ($model_name)
# and evaluates it using the same `demo_clean` setting ($task_config).

# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized pi0_base_aloha_robotwin_full demo_clean 0 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: Pi0.5

> Source: [https://robotwin-platform.github.io/doc/usage/Pi05.html](https://robotwin-platform.github.io/doc/usage/Pi05.html)

# OpenPI

## 1. Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies,you can add uv your conda environment.

```
conda activate RoboTwin
# Install uv
pip install uv
```

Once uv is installed, run the following commands to set up the environment:

```
cd policy/pi05
# Install prequisites in uv environment
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 1.1 IMPORTANT!!!

if error occured while build `av`, you should update `ffmpeg`, checking version by running:

```
ffmpeg -version
```

ffmpeg==n7.1 is already tested, you could install `ffmpeg` fllowing under command:

```
cd ~
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout n7.1
git pull origin n7.1

./configure --prefix="$HOME/ffmpeg-7.1-build" \
  --enable-gpl --enable-nonfree --enable-libx264 --enable-libx265 \
  --enable-libfdk-aac --enable-libmp3lame --enable-libopus \
  --enable-libvpx --enable-libass --enable-libfreetype \
  --enable-shared
make -j$(nproc)
make install

echo 'export PATH="$HOME/ffmpeg-7.1-build/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
# checkout if link success, should be n7.1
ffmpeg -version

sudo ln -s /home/xspark-ai/ffmpeg-5.1-build/include/* /usr/local/include/
sudo ln -s /home/xspark-ai/ffmpeg-5.1-build/lib/* /usr/local/lib/
sudo ldconfig
```

If you want to eval pi05 policy in RoboTwin，you are required to install curobo in your uv environment：

```
conda deactivate
source .venv/bin/activate
# At this point, you should be in the (openpi) environment
cd ../../envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../../policy/pi05/
bash
```

## 2. Generate RoboTwin Data

See [RoboTwin Tutorial (Usage Section)](https://robotwin-platform.github.io/doc/usage/collect-data.html) for more details.

## 3. Generate openpi Data

First, create the `processed_data` and `training_data` folders in the `policy/pi0` directory:

```
mkdir processed_data && mkdir training_data
```

Then, convert RoboTwin data to HDF5 data type.

```
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data_pi0.sh beat_block_hammer demo_clean 50
# or processing randomized data: bash process_data.sh beat_block_hammer demo_randomized 50
```

If success, you will find the `${task_name}-${task_config}-${expert_data_num}` folder under `policy/pi0/processed_data`.

Example folder structure:

```
processed_data/ 
├──${task_name}-${task_config}-${expert_data_num}
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...
```

Copy all the data you wish to use for training from `processed_data` into `training_data/${model_name}`. If you have multiple tasks with different data, simply copy them in the same way.please Place the corresponding task folders according to the example below.

```
#multi-task dataset example
training_data/  
├── ${model_name}
|       ├──${task_0}
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...
|       ├── ${task_1}
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...

#sigle task example
training_data/  
├── demo_clean
|       ├──beat_block_hammer-demo_clean-50
|       |   ├──episode_0
|       |   |   ├── instructions.json  
|       |   |   ├── episode_0.hdf5  
|       |   ├── episode_1 
|       |   |   ├── instructions.json  
|       |   |   ├── episode_1.hdf5  
|       |   ├── ...
```

Before generating the LerobotDataset format data for pi0,please make sure you have enough disk space under the `~/.cache`.This is because generating the `lerobotdataset` will require a large amount of space.And the datasets will be writed into `$XDG_CACHE_HOME`,which default path is `~/.cache`.If you don't have enough disk space under the `~/.cache` path, please use the following command to set a different cache directory with sufficient space:

```
export XDG_CACHE_HOME=/path/to/your/cache
```

Now, we can directly generate the LerobotDataset format data for pi0

```
# hdf5_path: The path to the generated HDF5 data (e.g., ./training_data/${model_name}/)
# repo_id: The name of the dataset (e.g., my_repo)
bash generate.sh ${hdf5_path} ${repo_id}
#bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

LerobotDataset format data will be writed into `${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}`

## 4. Write the Corresponding `train_config`

In `src/openpi/training/config.py`, there is a dictionary called `_CONFIGS`. You can modify 4 pre-configured PI0 and 1 pre-configured PI05 configurations I’ve written: `pi0_base_aloha_robotwin_lora` `pi0_fast_aloha_robotwin_lora` `pi0_base_aloha_robotwin_full` `pi0_fast_aloha_robotwin_full` `pi05_aloha_full_base`

You only need to write `repo_id` on your datasets.(e.g., `repo_id=demo_clean_repo`) If you want to change the `name` in `TrainConfig`, please include `fast` if you choose `pi_fast_base` model. If your do not have enough gpu memory, you can set `fsdp_devices`, refer to `config.py` line `src/openpi/training/config.py` line 526.

## 5. 5. Finetune model

```
# compute norm_stat for dataset
uv run scripts/compute_norm_stats.py --config-name ${train_config_name}
# uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base

# train_config_name: The name corresponding to the config in _CONFIGS, such as pi05_aloha_full_base
# model_name: You can choose any name for your model
# gpu_use: if not using multi gpu,set to gpu_id like 0;else set like 0,1,2,3
bash finetune.sh ${train_config_name} ${model_name} ${gpu_use}
#bash finetune.sh pi05_aloha_full_base demo_clean 0,1,2,3
```

| Training mode | Memory Required | Example GPU |
| --- | --- | --- |
| Fine-Tuning (LoRA) | > 46 GB | A6000(48G) |
| Fine-Tuning (Full) | > 100 GB | 2\*A100 (80GB) / 2\*H100 |

If your GPU memory is insufficient, please set the `fsdp_devices` parameter according to the following GPU memory reference, or reduce the `batch_size` parameter. Or you can try setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` in `finetune.sh`, it will cost lower gpu memory, but make training speed slower.

The default `batch_size` is 32 in the table below.

| GPU memory | Model type | GPU num | fsdp\_devices | Example GPU |
| --- | --- | --- | --- | --- |
| 24G | lora | 2 | 2 | 4090(24G) |
| 40G | lora | 2 | 2 | A100(40G) |
| 48G | lora | 1 | 1 | A6000(48G) |
| 40G | full | 4 | 4 | A100(40G) |
| 80G | full | 2 | 2 | A100(80G) |

## 6. Eval on RoboTwin

Checkpoints will be saved in policy/pi0/checkpoints/\({train\_config\_name}/\)}/${checkpoint\_id

You can modify the `deploy_policy.yml` file to change the `checkpoint_id` you want to evaluate.

```
# ckpt_path like: policy/pi0/checkpoints/pi0_base_aloha_robotwin_full/demo_clean/30000
bash eval.sh ${task_name} ${task_config} ${train_config_name} ${model_name} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_full demo_clean 0 0
# This command trains the policy using the `demo_clean` setting ($model_name)
# and evaluates it using the same `demo_clean` setting ($task_config).

# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized pi0_base_aloha_robotwin_full demo_clean 0 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: DexVLA

> Source: [https://robotwin-platform.github.io/doc/usage/DexVLA.html](https://robotwin-platform.github.io/doc/usage/DexVLA.html)

# DexVLA (Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning)

> Contributed by Midea Group

## 1. Install

To guarantee clean isolation between training and evaluation environments for both DexVLA and TinyVLA, we provide two distinct, self-contained setups.The training and testing environment can be used for both DexVLA and TinyVLA.

Training Environment：

```
cd policy/DexVLA
conda env create -f Train_Tiny_DexVLA_train.yml
conda activate dexvla-robo
cd policy_heads
pip install -e .
```

Evaluation Environment:

If you already have RoboTwin 2.0 installed, activate this conda environment and add the evaluation dependencies:

```
conda activate your_RoboTwin_env
pip install -r Eval_Tiny_DexVLA_requirements.txt
```

## 2. Prepare Training Data

This step performs data preprocessing, converting the original RoboTwin 2.0 data into the format required for DexVLA training. The `expert_data_num` parameter specifies the number of trajectory pairs to be used as training data.

```
python process_data.py ${task_name} ${task_config} ${expert_data_num}
# python process_data.py beat_block_hammer demo_clean 50
```

If success, you will find the data in the `policy/Dexvla/data/sim_${task_name}/${setting}_${expert_data_num}` folder.

## 3. Train Policy

This step launches the training process.

### 3.1 Download official Qwen2\_VL weights

We construct the VLM backbone by integrating Qwen2-VL-2B.You can download the official weights from this link:

| Model | Link |
| --- | --- |
| Qwen2-VL (~2B) | [huggingface](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |

**❗❗** After downloading the standard weights, you have to modify the official `config.json` file in the folder. Please update the 'architectures' field from "Qwen2VLForConditionalGenerationForVLA" to "DexVLA", and change the 'model\_type' field from "qwen2\_vla" to "dex\_vla".

### 3.2 Download our pretrained ScaleDP-H weights

We released our pretrained weights of ScaleDP-H which is trained after Stage1. Now you can download the weights and directly finetuning your data on Stage 2.

| Model | Link |
| --- | --- |
| ScaleDP-H (~1B) | [huggingface](https://huggingface.co/lesjie/scale_dp_h) |
| ScaleDP-L (~400M) | [huggingface](https://huggingface.co/lesjie/scale_dp_l) |
| ### 3.3 Train |  |
| The training script are "scripts/aloha/vla\_stage2\_train.sh". And you need to change following parameters: |  |
| 1. **OUTPUT** : refers to the save directory for training, which must include the keyword "qwen2" (and optionally "lora"). If LoRA training is used, the name must include "lora" (e.g., "qwen2\_lora"). |  |
| 2. **TASKNAME** : refers to the tasks used for training, which should be corresponded to "your\_task\_name" in aloha\_scripts/constant.py |  |
| 3. **mnop** : path to the pretrained VLM weights |  |
| 4. **load\_pretrain\_dit** : True |  |
| 5. **DIT\_PRETRAIN** :Path to pretrained policy head (ScaleDP). |  |

Other hyperparameters like "batch\_size", "save\_steps" could be customized according to your computation resources.

Start training by following commands:

```
bash ./scripts/aloha/vla_stage2_train.sh
```

## 4. Eval Policy

You need to modify the corresponding path in the `deploy_policy.yml` file: 1. **model\_path** : Path to the trained model, in the OUTPUT path. 2. **state\_path** : Path to `dataset_stats.pkl`, in the OUTPUT path.

Then execute:

```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean demo_clean 0 50 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 0 50 0 0
```

## 5. Citation

If you find our works useful for your research and applications, please cite using these BibTeX:

### 5.1 DexVLA

```
@article{wen2025dexvla,
  title={DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control},
  author={Wen, Junjie and Zhu, Yichen and Li, Jinming and Tang, Zhibin and Shen, Chaomin and Feng, Feifei},
  journal={arXiv preprint arXiv:2502.05855},
  year={2025}
}
```

### 5.2 DiffusionVLA

```
@article{wen2024diffusion,
  title={Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression},
  author={Wen, Junjie and Zhu, Minjie and Zhu, Yichen and Tang, Zhibin and Li, Jinming and Zhou, Zhongyi and Li, Chengmeng and Liu, Xiaoyu and Peng, Yaxin and Shen, Chaomin and others},
  journal={arXiv preprint arXiv:2412.03293},
  year={2024}
}
```

### 5.3 ScaleDP

```
@article{zhu2024scaling,
  title={Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation},
  author={Zhu, Minjie and Zhu, Yichen and Li, Jinming and Wen, Junjie and Xu, Zhiyuan and Liu, Ning and Cheng, Ran and Shen, Chaomin and Peng, Yaxin and Feng, Feifei and others},
  journal={arXiv preprint arXiv:2409.14411},
  year={2024}
}
```


---

## Page: TinyVLA

> Source: [https://robotwin-platform.github.io/doc/usage/TinyVLA.html](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)

# Tiny-VLA (Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation)

> Contributed by Midea Group

## 1. Install

To guarantee clean isolation between training and evaluation environments for both DexVLA and TinyVLA, we provide two distinct, self-contained setups.The training and testing environment can be used for both DexVLA and TinyVLA.

Training Environment：

```
cd policy/TinyVLA
conda env create -f Train_Tiny_DexVLA_train.yml
conda activate dexvla-robo
cd policy_heads
pip install -e .
```

Evaluation Environment:

If you already have RoboTwin 2.0 installed, activate its conda environment and add the evaluation dependencies:

```
conda activate your_RoboTwin_env
pip install -r Eval_Tiny_DexVLA_requirements.txt
```

## 2. Prepare Training Data

This step performs data preprocessing, converting the original RoboTwin 2.0 data into the format required for TinyVLA training. The `expert_data_num` parameter specifies the number of trajectory pairs to be used as training data.

```
python process_data.py ${task_name} ${task_config} ${expert_data_num}
# python process_data.py beat_block_hammer demo_randomized 50
```

If success, you will find the `sim_${task_name}/${setting}_${expert_data_num}` folder under `policy/Tinyvla/data`.

## 3. Train Policy

This step launches the training process. First, download the VLM model InternVL3-1B ([huggingface](https://huggingface.co/OpenGVLab/InternVL3-1B/tree/main)) to the path `.../policy/TinyVLA/model_param/InternVL3-1B`. Then modify the `config.json` file in the folder as follows:

```
{
    "_name_or_path": ".../robotiwin/policy/TinyVLA/vla/models/internvl", # Modify this.
    "architectures": [
        "TinyVLA" # Change this.
    ],
    # "auto_map":{...} # Delete this.
    ...
    "llm_config": {}, # Don't Change.
    "min_dynamic_patch": 1,
    "model_type": "tinyvla", # Change this.
    ...
}
```

Then add an task config item in `.../policy/TinyVLA/aloha_scripts/constants.py`

```
TASK_CONFIGS = {
    ...
    "your_task": {
        'dataset_dir': [DATA_DIR + "/sim-your_task/aloha-agilex-1-m1_b1_l1_h0.03_c0_D435-100"],
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    }
}
```

Then begin the training

```
bash ./scripts/franks/train_robotwin_aloha.sh
```

Configure the training by modifying the following items in the `train_robotwin_aloha.sh` file.

```
TASK=your_task # Set the Task
ROOT=.../robotiwin/policy/TinyVLA # Set Root Path
mnop=.../robotiwin/policy/TinyVLA/model_param/InternVL3-1B/ # Set The Path of base VLM
```

## 4. Eval Policy

You need to modify the corresponding path in the `deploy_policy.yml` file: 1. **model\_path** : Path to the trained model, in the OUTPUT path. 2. **state\_path** : Path to `dataset_stats.pkl`, in the OUTPUT path. 3. **model\_base** : Path to InternVL3-1B.

Then execute:

```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_randomized 0 50 0 0
```

## 5. Citation

If you find Tiny-VLA useful for your research and applications, please cite using this BibTeX:

```
@inproceedings{wen2024tinyvla,
    title={Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation},
    author={Wen, Junjie and Zhu, Yichen and Li, Jinming and Zhu, Minjie and Wu, Kun and Xu, Zhiyuan and Liu, Ning and Cheng, Ran and Shen, Chaomin and Peng, Yaxin and others},
    booktitle={IEEE Robotics and Automation Letters (RA-L)},
    year={2025}
}
```


---

## Page: OpenVLA-oft

> Source: [https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)

# Openvla-oft

## 1. Environment Setup

The conda environment for openvla-oft with RoboTwin is identical to the official openvla-oft environment for the ALOHA part. Please follow the ([openvla-oft official documentation](https://github.com/moojink/openvla-oft/blob/main/SETUP.md)) to install the environment and directly overwrite the RoboTwin environment.

```
conda activate RoboTwin
# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

**Note!**  
 If you encounter problems on diffusers, try `pip install diffusers==0.33.1`

## 2. Collect RoboTwin Data

See [RoboTwin Tutorial (Usage Section)](https://robotwin-platform.github.io/doc/usage/collect-data.html) for more details.

## 3. Generate RLDS Data

> RLDS dataset is the data format required for Openvla-oft training.

use RoboTwin data generation mechanism to generate data.   
 Then convert the raw data to the aloha format that openvla-oft accepts:

```
bash preprocess_aloha.sh
```

Then transform the data to tfds form and register the tfds form dataset in your device: e.g.:

```
python -m datasets.move_can_pot_builder
```

After converting to RLDS, register the dataset (which, for example, would be called `aloha_move_can_pot_builder`) with our dataloader by adding an entry for it in `configs.py` ([here](https://robotwin-platform.github.io/doc/usage/prismatic/vla/datasets/rlds/oxe/configs.py#L680)), `transforms.py` ([here](https://robotwin-platform.github.io/doc/usage/prismatic/vla/datasets/rlds/oxe/transforms.py#L928)), and `mixtures.py` ([here](https://robotwin-platform.github.io/doc/usage/prismatic/vla/datasets/rlds/oxe/mixtures.py#L216)).Details in [Openvla-oft official documentation](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md)

## 4. Finetune model

```
bash finetune_aloha.sh
```

By default, the training process will not save merged weights. So you need to run `merge_lora.sh` to merge lora weights if you want to use the checkpoint. If some `.py` files miss in the merged checkpoint, just copy them from the original checkpoint.

## 5. Eval on RoboTwin

example usage

```
bash eval.sh ${task_name} ${task_config} ${checkpoint_path} ${seed} ${gpu_id} ${unnorm_key}
# Example: bash eval.sh move_can_pot demo_randomized ckpt_path 0 5 aloha_move_can_pot_builder
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.


---

## Page: LLaVA-VLA

> Source: [https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)

# LLaVA-VLA

> Contributed by IRPN Lab, HKUST(GZ)  
>  Email: songwenxuan0115@gmail.com, sunxiaoquan@hust.edu.cn

## 1. Environment Setup

See [LLaVA-VLA installation](https://github.com/OpenHelix-Team/LLaVA-VLA?tab=readme-ov-file#installation) for more details.

## 2. Download Model

Please download the corresponding model from the [model zoo](https://github.com/OpenHelix-Team/LLaVA-VLA?tab=readme-ov-file#modelzoo).

## 3. Collect RoboTwin Data

See [RoboTwin Tutorial (Usage Section)](https://robotwin-platform.github.io/doc/usage/collect-data.html#1-environment-setup) for more details.

## 4. Generate Image and Data

First, create the pictures folder in the policy/LLaVA-VLA directory:

```
mkdir pictures && training_data
cd scripts && cd helper
```

Then, extract the original image from RoboTwin data.

```
bash image_extraction.sh ${task_name} ${task_config}
# bash image_extraction.sh grab_roller demo_randomized
# bash image_extraction.sh all demo_randomized
# In task_name, you can directly select a task(such as: grab_roller) or choose "all" (just modify it in task_list).
```

Next, generate the format data required for LLaVA-VLA training.

```
bash process_data.sh ${task_name} ${task_config} ${future_chunk}
# bash process_data.sh grab_roller demo_randomized 5
# bash process_data.sh all demo_randomized 5
# In task_name, you can directly select a task(such as: grab_roller) or choose "all" (just modify it in task_list). 
# future_chunk: The number of output steps in the future (default is 5).
```

Example folder structure:

```
training_data
├── ${task_1}
│   ├── ${task_config_1}
|   |   |── episode0.json
|   |   |── episode1.json
│   ├── ${task_config_2}
|   |   |── episode0.json
|   |   |── episode1.json
├── ${task_2}
│   ├── ...
├── ...
```

```
pictures
├── ${task_1}
│   ├── ${task_config_1}
|   |   |── episode0
|   |   |   |── 01.jpg
|   |   |   |── 02.jpg
│   ├── ${task_config_2}
|   |   |── episode0
|   |   |   |── 01.jpg
|   |   |   |── ...
├── ${task_2}
│   ├── ...
├── ...
```

## 5. merge json and Generate yaml file

In this step, we need to merge all the JSON files generated by the previous `process_data` step into a single JSON file.

```
python llava/process_data/merge_json.py
# please replace `yourpath` with your actual path!
```

```
python llava/process_data/yaml_general.py
```

## 6. Pre-Training

Before starting the training, please replace `yourpath` with your actual path!

```
bash calvin_finetune_obs.sh
```

## 7. Fine-tuning

Please note to change `MODEL_NAME_OR_PATH` to the checkpoint generated in the previous step. For the dataset you fine-tuned, please regenerate the `ACTION_STAT` file and modify `JSON_PATH`.Then

```
bash calvin_finetune_obs.sh
```

## 8. Eval on RoboTwin

You need to modify the corresponding path in the deploy\_policy.yml file: 1. `model_path` : Path to the checkpoint. 2. `action_stat` : Path to dataset\_statistic.yaml.

```
bash eval.sh ${gpu_id}
# bash eval.sh 0
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.

## 9. Citation

If you find our works useful for your research and applications, please cite using these BibTeX:

```
@article{pdvla,
  title={Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding},
  author={Song, Wenxuan and Chen, Jiayi and Ding, Pengxiang and Zhao, Han and Zhao, Wei and Zhong, Zhide and Ge, Zongyuan and Ma, Jun and Li, Haoang},
  journal={arXiv preprint arXiv:2503.02310},
  year={2025}
}
```


---

## Page: GO1

> Source: [https://robotwin-platform.github.io/doc/usage/GO1.html](https://robotwin-platform.github.io/doc/usage/GO1.html)

# GO-1 Fine-tuning and Evaluation

> Contributed by GO-1 Team

This README provides instructions for fine-tuning and evaluating GO-1 model, including data generation, processing, model training, and evaluation.

## 1. Table of Contents

- [GO-1 Fine-tuning and Evaluation](#go-1-fine-tuning-and-evaluation)
- [Table of Contents](#1-table-of-contents)
- [Environment Setup](#2-environment-setup)
  - [1. Install RoboTwin](#21-1-install-robotwin)
  - [2. Install GO-1](#2-install-go-1)
- [Data Generation](#3-data-generation)
- [Data Processing](#4-data-processing)
  - [1. Convert RoboTwin Data to HDF5](#41-1-convert-robotwin-data-to-hdf5)
  - [2. Convert HDF5 to LeRobot Dataset](#42-2-convert-hdf5-to-lerobot-dataset)
- [Model Fine-tuning](#model-fine-tuning)
- [Evaluation](#6-evaluation)
  - [Start GO-1 Server](#start-go-1-server)
  - [Start RoboTwin Client](#62-start-robotwin-client)
- [Evaluation Results](#6-evaluation-results)

## 2. Environment Setup

### 2.1 1. Install RoboTwin

Create the conda environment and install the dependencies for RoboTwin according to the [RoboTwin docs](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

Then install the extra dependencies:

```
cd policy/GO1

conda activate RoboTwin
pip install -r requirements.txt
```

### 2.2 2. Install GO-1

Follow the instructions in the [GO-1 repo](https://github.com/OpenDriveLab/AgiBot-World?tab=readme-ov-file#getting-started--) to set up a **separate** conda environment for GO-1.

## 3. Data Generation

Follow the [RoboTwin docs](https://robotwin-platform.github.io/doc/usage/collect-data.html) to generate raw data in RoboTwin format.

Your raw data should be organized as follows:

```
data/
├── task_name/
│   ├── task_config/
│   │   ├── data/
│   │   │   ├── episode0.hdf5
│   │   │   ├── episode1.hdf5
│   │   │   └── ...
│   │   └── instructions/
│   │       ├── episode0.json
│   │       ├── episode1.json
│   │       └── ...
```

## 4. Data Processing

### 4.1 1. Convert RoboTwin Data to HDF5

```
# Activate the RoboTwin environment
conda activate RoboTwin

bash robotwin2hdf5.sh <task_name> <task_config> <expert_data_num>

# Example:
bash robotwin2hdf5.sh beat_block_hammer demo_clean 50
```

This will create processed data in the `processed_data/<task_name>-<task_config>-<expert_data_num>` directory.

### 4.2 2. Convert HDF5 to LeRobot Dataset

```
# Activate the GO-1 environment
conda activate go1

# Optional: Change the LeRobot home directory
export HF_LEROBOT_HOME=/path/to/your/lerobot

bash hdf52lerobot.sh <hdf5_path> <repo_id>

# Example:
bash hdf52lerobot.sh processed_data/beat_block_hammer-demo_clean-50/ beat_block_hammer_repo
```

The LeRobot dataset will be saved in `<HF_LEROBOT_HOME>/<repo_id>`.

## 5. Model Fine-tuning

Refer to the [GO-1 repo](https://github.com/OpenDriveLab/AgiBot-World?tab=readme-ov-file#fine-tuning-on-your-own-dataset-) for detailed instructions.

## 6. Evaluation

### 6.1 Start GO-1 Server

Start the GO-1 inference server using your fine-tuned model checkpoint and data statistics:

```
cd /path/to/AgiBot-World

conda activate go1

python evaluate/deploy.py --model_path /path/to/your/checkpoint --data_stats_path /path/to/your/dataset_stats.json --port <SERVER_PORT>
```

The server will will listen on port `SERVER_PORT` and wait for observations.

### 6.2 Start RoboTwin Client

The client requires a separate terminal session. We strongly recommend using `tmux` or `screen` for this process, as evaluation can take several hours to complete.

First config the client in [deploy\_policy.yml](https://robotwin-platform.github.io/doc/usage/deploy_policy.yml):

```
host: Server IP address (default: 127.0.0.1)
port: Server port (default: 9000)
```

Then use the provided [script](https://robotwin-platform.github.io/doc/usage/eval.sh) to evaluate your model:

```
conda activate RoboTwin

bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>

# Example:
bash eval.sh beat_block_hammer demo_clean go1_demo 0 0
```

**Arguments:** - `task_name` - Name of the task (*e.g.*, `beat_block_hammer`) - `task_config` - Task configuration (*e.g.*, `demo_randomized`, `demo_clean`) - `ckpt_setting` - Checkpoint setting name (default: `go1_demo`) - `seed` - Random seed (default: `0`) - `gpu_id` - GPU ID to use (default: `0`)

Alternatively, you can set these values in [deploy\_policy.yml](https://robotwin-platform.github.io/doc/usage/deploy_policy.yml).

The evaluation results, including videos and metrics, will be saved in the `eval_result/<task_name>/GO1/<task_config>/<ckpt_setting>` directory under the project root.

## 7. Evaluation Results

Following the setup in [RoboTwin2.0 Benchmark](https://robotwin-platform.github.io/leaderboard), we report the performance of GO-1 Air model and other baselines in the table below. All models are trained on the Aloha-AgileX embodiment using 50 `demo_clean` demonstrations for 3 selected tasks (`grab_roller`, `handover_mic`, `lift_pot`), and evaluated 100 times under the `demo_clean (Easy)` and `demo_randomized (Hard)` settings. Our models are fine-tuned for 10k steps.

| Policy Task | Grab Roller |  | Lift Pot |  | Average |
| --- | --- | --- | --- | --- | --- |
|  | Easy | Hard | Easy | Hard |  |
| DP | 98% | 0% | 39% | 0% | 34.25% |
| ACT | 94% | 25% | 88% | 0% | 51.25% |
| RDT | 74% | 43% | 72% | 9% | 49.5% |
| Pi0 | **96%** | 80% | 84% | **36%** | 74% |
| GO-1 Air | 86% | 94% | **94%** | 33% | 76.75% |
| GO-1 | **96%** | **96%** | **94%** | 35% | **80.25%** |


---

## Page: Expert Code Gen (for Novel Task)

> Source: [https://robotwin-platform.github.io/doc/usage/expert-code-gen.html](https://robotwin-platform.github.io/doc/usage/expert-code-gen.html)

# Expert Code Generation

## 1. Code\_gen Folder Structure

This directory contains various modules for generating and testing robot task code:

- **gpt\_agent.py**: API integration with LLM models
- **observation\_agent.py**: Processes multi-modal observations for code correction
- **prompt.py**: Prompt templates for code generation
- **run\_code.py**: Executes and tests generated code
- **task\_generation\_simple.py**: Basic single-pass code generation
- **task\_generation.py**: Iterative code generation with error feedback
- **task\_generation\_mm.py**: Advanced code generation with multi-modal observation
- **task\_info.py**: Task definitions and descriptions
- **test\_gen\_code.py**: Utility for testing generated code with detailed metrics

The code generation system also interacts with these important directories: - **./envs/**: Contains manually implemented task environments - **\_base\_task.py**: Core environment with robot control functions and utilities - Includes `save_camera_images(task_name, step_name, generate_num_id, save_dir)` for capturing visual observations during task execution - **./envs\_gen/**: Stores auto-generated task implementations - **./task\_config/**: Configuration files for tasks and embodiments - **./script/**: Template scripts and utilities - **./assets/objects/**: 3D models and metadata for simulation objects - **./camera\_images/**: Stores observation images captured during code generation for multi-modal feedback

The entire pipeline enables automatic generation of robot control code from natural language task descriptions, with feedback-based refinement and multi-modal observation capabilities.

## 2. Configure LLM API Key

Please configure the necessary API keys in the `code_gen/gpt_agent.py` file. Additionally, if the LLM you are utilizing does not support integration with the OpenAI API, you may need to make corresponding adjustments to the `generate()` function.

## 3. Generate Your Task Code

### 3.1 1. **Add Task Description**

Add new task information, including the task name and natural language description, in `./code_gen/task_info.py`.

# 1. Template of Task Information:

```
TASK_NAME = {
    "task_name": "task_name",                # Name of the task
    "task_description": "...",               # Detailed description of the task
    "current_code": '''
                class gpt_{task_name}({task_name}):
                    def play_once(self):
                        pass
                '''                          # Code template to be completed
    "actor_list": {                          # List of involved objects; can be a dictionary or a simple list
        "self.object1": {
            "name": "object1",               # Object name
            "description": "...",            # Description of the object
            "modelname": "model_name"        # Name of the 3D model representing the object
        },
        "self.object2": {
            "name": "object2",
            "description": "...",
            "modelname": "model_name"
        },
        # ... more objects
    },
    # Alternatively, the actor_list can be a simple list:
    # "actor_list": ["self.object1", "self.object2", ...],
    # To make code generation easier, the actor_list also includes some pose information
    # like target poses or middle poses (optional and don't require modelname).
}
```

### 1.1 2. **Add Basic Task Code**

Add the basic code file `${task_name}.py` in the `./envs/` directory, following this structure:

```
from .base_task import Base_task
from .utils import *
import sapien

class ${task_name}(Base_task):
    def setup_demo(self, **kwargs):
        # Initializes the simulation environment for the task
        # Sets up the table, robot, planner, camera, and initial positions
        # This function is called once at the beginning of each episode
        pass

    def load_actors(self):
        # Loads all the necessary objects for the task into the environment
        # Typically called from setup_demo to initialize scene objects
        # Can also be used to set initial poses for objects
        pass

    def play_once(self):
        # Contains the robot control code to complete the task
        # This is the main function that will be generated by the LLM
        # Implements the sequence of actions for the robot to achieve the task
        pass

    # Check success
    def check_success(self):
        # Defines criteria to determine if the task was completed successfully
        # Returns a boolean indicating success or failure
        # Used for evaluation and feedback during code generation
        pass
```

In the code above, `{task_name}` should match the name of the basic code file, and the `check_success()` function is used to determine if the task is successful. No changes are needed for the rest of the code.

> Note: The `envs` folder contains manually written files with `setup_demo`, robot operation code in `play_once`, and `check_success` methods. Auto-generated code will be saved in the `envs_gen` folder.

### 1.2 3. **Generate the Final Code**

You can use three different code generation approaches depending on your needs:

**Note: The code generation process will only generate the `play_once()` method implementation, which contains the robot control logic to complete the task. Other methods like `setup_demo()`, `load_actors()`, and `check_success()` should be manually implemented.**

#### 1.2.1 Basic Code Generation

For quick verification of new tasks or debugging existing ones without iterative correction:

```
python code_gen/task_generation_simple.py task_name
```

#### 1.2.2 Code Generation with Error Feedback

This script implements iterative code correction based on error feedback, consistent with RoboTwin 1.0:

```
python code_gen/task_generation.py task_name
```

#### 1.2.3 Advanced Code Generation with Multi-Modal Observations

This script provides both error feedback iteration and multi-modal observation-based code correction, consistent with RoboTwin 2.0. It offers the best generation quality but runs slower:

```
python code_gen/task_generation_mm.py task_name
```

The multi-modal observation functionality is implemented in `code_gen/observation_agent.py`.

The generated code file will be saved as `./envs_gen/gpt_${task_name}.py`. For example:

```
python code_gen/task_generation_mm.py pick_dual_bottles_easy
```

This will create `./envs_gen/gpt_pick_dual_bottles_easy.py`.

### 1.3 4. **Test Generated Code**

Run the following script to test the generated code:

```
python code_gen/run_code.py task_name
```

This will execute the task using the generated code and display the results, allowing you to validate the performance.

## 1.1 Additional Resources

For more information on generating task descriptions and object descriptions, refer to the documentation in the [description](https://robotwin-platform.github.io/doc/usage/../description/README.md) directory.

For policy training and evaluation using the generated code, consult the [policy/ACT](https://robotwin-platform.github.io/doc/usage/../policy/ACT/README.md) documentation.


---

## Page: API Tutorial

> Source: [https://robotwin-platform.github.io/doc/usage/API.html](https://robotwin-platform.github.io/doc/usage/API.html)

# API for Controlling Mechanical Arms

The API can be used to control one or two robotic arms to perform operations such as grasping, placing, moving, and returning to the origin. Each arm is identified by an `ArmTag`, which can be `"left"` or `"right"`. Actions are generated in sequences and executed together via the `move()` method.

---

## 1. Class Structure

- **`self`**: The task class inherit from `Base_Task`.
- **`ArmTag`**: A custom type representing a robotic arm. It supports comparison with strings: `ArmTag("left") == "left"` returns `True`. You can obtain the opposite arm using `ArmTag("left").opposite`, i.e., `ArmTag("left").opposite == "right"` returns `True`.
- **`Actor`**/**`ArticulationActor`**: The object being manipulated. Provides methods to retrieve key points (contact point `contact_point`, functional point `functional_point`, target point `target_point`) and its current global pose.
- **`Action`**: A sequence of actions for controlling the arm. You only need to know that it can be executed via the `move()` function.

---

## 2. Controlling APIs

### 2.1 `move(self, actions_by_arm1: tuple[ArmTag, list[Action]], actions_by_arm2: tuple[ArmTag, list[Action]] = None)`

#### 2.1.1 Description

Executes action sequences on one or both robotic arms simultaneously.

#### 2.1.2 Parameters

- `actions_by_arm1`: Action sequence for the first arm, formatted as `(arm_tag, [action1, action2, ...])`
- `actions_by_arm2`: Optional, action sequence for the second arm

#### 2.1.3 Notes

- The same `ArmTag` cannot be passed twice.
- All actions must have been pre-generated.

#### 2.1.4 Example

One arm grasps a bottle, the other moves back to avoid interference.

```
self.move(
    self.grasp_actor(self.bottle, arm_tag=arm_tag),
    self.back_to_origin(arm_tag=arm_tag.opposite)
)
```

---

### 2.2 `grasp_actor(self, actor: Actor, arm_tag: ArmTag, pre_grasp_dis=0.1, grasp_dis=0, gripper_pos=0., contact_point_id=None) -> tuple[ArmTag, list[Action]]`

#### 2.2.1 Description

Generates a sequence of actions to pick up the specified `Actor`.

#### 2.2.2 Parameters

- `actor`: The object to grasp
- `arm_tag`: Which arm to use
- `pre_grasp_dis`: Pre-grasp distance (default 0.1 meters), the arm will move to this position first
- `grasp_dis`: Grasping distance (default 0 meters), the arm moves from the pre-grasp position to this position and then closes the gripper
- `gripper_pos`: Gripper closing position (default 0, fully closed)
- `contact_point_id`: Optional list of contact point IDs; if not provided, the best grasping point is selected automatically

#### 2.2.3 Returns

`(arm_tag, action_list)` containing the grasp actions.

#### 2.2.4 Example

Select appropriate grasp point based on arm\_tag and grasp the cup.

```
self.move(
    self.grasp_actor(
        self.cup, arm_tag=arm_tag,
        pre_grasp_dis=0.1,
        contact_point_id=[0, 2][int(arm_tag=='left')]
    )
)
```

---

### 2.3 `place_actor(self, actor: Actor, arm_tag: ArmTag, target_pose: list | np.ndarray, functional_point_id: int = None, pre_dis=0.1, dis=0.02, is_open=True, **kwargs) -> tuple[ArmTag, list[Action]]`

#### 2.3.1 Description

Places a currently held object at a specified target pose.

#### 2.3.2 Parameters

- `actor`: The currently held object
- `arm_tag`: The arm holding the object
- `target_pose`: Target position/orientation, length 3 or 7 (xyz + optional quaternion)
- `functional_point_id`: Optional ID of the functional point; if provided, aligns this point to the target, otherwise aligns the base of the object
- `pre_dis`: Pre-place distance (default 0.1 meters), arm moves to this position first
- `dis`: Final placement distance (default 0.02 meters), arm moves from pre-place to this location, then opens the gripper
- `is_open`: Whether to open the gripper after placing (default True)
- `**kwargs`: Other optional parameters:
  - `constrain : {'free', 'align', 'auto'}, default='auto'` Alignment strategy:
    - `free`: Only forces the object's z-axis to align with the target point's z-axis, other axes are determined by projection.
    - `align`: Forces all axes of the object to align with all axes of the target point.
    - `auto`: Automatically selects a suitable placement pose based on grasp direction (vertical or horizontal).
  - `align_axis : list of np.ndarray or np.ndarray or list, optional` Vectors or vector list in world coordinates to align with. For example, `[1, 0, 0]` or `[[1, 0, 0], [0, 1, 0]]`. If multiple vectors are provided, the one with the smallest dot product with the current actor axis will be chosen for alignment.
  - `actor_axis : np.ndarray or list, default=[1, 0, 0]` The second object axis used for alignment (the first is the z-axis which will be forced to align). Typically used for auxiliary alignment (especially when `constrain == 'align'`).
  - `actor_axis_type : {'actor', 'world'}, default='actor'` Specifies whether `actor_axis` is relative to the object coordinate system or world coordinate system.
  - `pre_dis_axis : {'grasp', 'fp'} or np.ndarray or list, default='grasp'` Specifies the pre-placement offset direction:
    - `grasp`: Offset along the grasp direction (i.e., opposite to the end-effector pointing towards the object center).
    - `fp`: Offset along the target point's z-axis direction.
    - Custom vectors can also be provided to represent the offset direction.

#### 2.3.3 Returns

`(arm_tag, action_list)` containing the place actions.

#### 2.3.4 Example

When stacking one object on top of another (for example, placing blockA on top of blockB).

```
target_pose = self.last_actor.get_functional_point(point_id, "pose")
# Use this target_pose in place_actor to place the object exactly on top of last_actor at the specified functional point.
self.move(
    self.place_actor(
        actor=self.current_actor, # The object to be placed
        target_pose=target_pose, # The pose acquired from last_actor
        arm_tag=arm_tag,
        functional_point_id=0, # Align functional point 0, or specify as needed
        pre_dis=0.1,
        dis=0.02,
        pre_dis_axis="fp", # Use functional point direction for pre-displacement, if the functional point is used
    )
)
```

Place the actor at actor\_pose (already a Pose object).

```
self.move(
    self.place_actor(
        self.box,
        target_pose=self.actor_pose, # already a Pose, no need for get_pose()
        arm_tag=grasp_arm_tag,
        functional_point_id=0, # functional_point_id can be retrived from the actor list if the actor has functional points
        pre_dis=0,
        dis=0,  # set dis to 0 if is_open is False, and the gripper will not open after placing. Set the `dis` to a small value like 0.02 if you want the gripper to open after placing.
        is_open=False, # if is_open is False, pre_dis and dis will be 0, and the gripper will not open after placing.
        constrain="free", # if task requires the object to be placed in a specific pose that mentioned in the task description (like "the head of the actor should be toward xxx), you can set constrain to "align", in all of other cases, you should set constrain to "free".
        pre_dis_axis='fp', # Use functional point direction for pre-displacement, if the functional_point_id is used
    )
)
```

---

### 2.4 `move_by_displacement(self, arm_tag: ArmTag, x=0., y=0., z=0., quat=None, move_axis='world') -> tuple[ArmTag, list[Action]]`

#### 2.4.1 Description

Moves the end-effector of the specified arm along relative directions and sets its orientation.

#### 2.4.2 Parameters

- `arm_tag`: The arm to control
- `x`, `y`, `z`: Displacement along each axis (in meters)
- `quat`: Optional quaternion specifying the target orientation; if not set, uses current orientation
- `move_axis`: `'world'` means displacement is in world coordinates, `'arm'` means displacement is in local coordinates

#### 2.4.3 Returns

`(arm_tag, action_list)` containing the move-by-displacement actions.

#### 2.4.4 Example

Lift the object up by moving relative to current position, you should lift the arm up evrery time after grasping an object to avoid collision.

```
self.move(
    self.move_by_displacement(
        arm_tag=arm_tag,
        z=0.07,  # Move 7cm upward
        move_axis='world'
    )
)
```

---

### 2.5 `move_to_pose(self, arm_tag: ArmTag, target_pose: list) -> tuple[ArmTag, list[Action]]`

#### 2.5.1 Description

Moves the end-effector of the specified arm to a specific absolute pose.

#### 2.5.2 Parameters

- `arm_tag`: The arm to control
- `target_pose`: Absolute position and/or orientation, length 3 or 7 (xyz + optional quaternion)

#### 2.5.3 Returns

`(arm_tag, action_list)` containing the move-to-pose actions.

#### 2.5.4 Example

Move the arm to a specific pose, for example, to place an object in a certain position decided by which arm is placing the object.

```
target_pose = self.get_arm_pose(arm_tag=arm_tag)
if arm_tag == 'left':
    # Set specific position and orientation for left arm
    target_pose[:2] = [-0.1, -0.05]
    target_pose[2] -= 0.05
    target_pose[3:] = [-0.707, 0, -0.707, 0]
else:
    # Set specific position and orientation for right arm
    target_pose[:2] = [0.1, -0.05]
    target_pose[2] -= 0.05
    target_pose[3:] = [0, 0.707, 0, -0.707]

# Move the skillet to the defined target pose
self.move(
    self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose)
)
```

---

### 2.6 `close_gripper(self, arm_tag: ArmTag, pos=0.) -> tuple[ArmTag, list[Action]]`

#### 2.6.1 Description

Closes the gripper of the specified arm.

#### 2.6.2 Parameters

- `arm_tag`: Which arm's gripper to close
- `pos`: Gripper position (0 = fully closed)

#### 2.6.3 Returns

`(arm_tag, action_list)` containing the gripper-close action.

#### 2.6.4 Example

```
self.move(
    self.close_gripper(arm_tag=arm_tag)
)
```

---

### 2.7 `open_gripper(self, arm_tag: ArmTag, pos=1.) -> tuple[ArmTag, list[Action]]`

#### 2.7.1 Description

Opens the gripper of the specified arm.

#### 2.7.2 Parameters

- `arm_tag`: Which arm's gripper to open
- `pos`: Gripper position (1 = fully open)

#### 2.7.3 Returns

`(arm_tag, action_list)` containing the gripper-open action.

#### 2.7.4 Example

```
self.move(
    self.open_gripper(arm_tag=arm_tag)
)
```

---

### 2.8 `back_to_origin(self, arm_tag: ArmTag) -> tuple[ArmTag, list[Action]]`

#### 2.8.1 Description

Returns the specified arm to its predefined initial position.

#### 2.8.2 Parameters

- `arm_tag`: The arm to return to origin

#### 2.8.3 Returns

`(arm_tag, action_list)` containing the return-to-origin action.

#### 2.8.4 Example

Place left object while moving right arm back to origin.

```
move_arm_tag = ArmTag("left")  # Specify which arm is placing the object
back_arm_tag = ArmTag("right")  # Specify which arm is moving back to origin
self.move(
    self.place_actor(
        actor=self.left_actor,
        arm_tag=move_arm_tag,
        target_pose=target_pose,
        pre_dis_axis="fp",
    ),
    self.back_to_origin(arm_tag=back_arm_tag)
)
```

---

### 2.9 `get_arm_pose(self, arm_tag: ArmTag) -> list[float]`

#### 2.9.1 Description

Gets the current pose of the end-effector of the specified arm.

#### 2.9.2 Parameters

- `arm_tag`: Which arm to query

#### 2.9.3 Returns

A list of 7 floats: `[x, y, z, qw, qx, qy, qz]`, representing position and orientation.

#### 2.9.4 Example

```
pose = self.get_arm_pose(ArmTag("left"))
```

---

## 3. `Actor` Class APIs

`Actor` is the object being manipulated by the robotic arms. It provides methods to retrieve key points and its current global pose. The `Actor` class has the following data points:

- Target Point `target_point`: Special points available during planning (e.g., handle of a cup)
- Contact Point `contact_point`: Position where the robotic arm grasps the object (e.g., rim of a cup)
- Functional Point `functional_point`: Position where the object interacts with other objects (e.g., head of a hammer)
- Orientation Point `orientation_point`: Specifies the orientation of the object (e.g., toe of a shoe pointing left)

These methods can be called on `Actor` objects:

### 3.1 `get_contact_point(self, idx: int) -> list[float]`

Returns the pose of the `idx`-th contact point as `[x, y, z, qw, qx, qy, qz]`

### 3.2 `get_functional_point(self, idx: int) -> list[float]`

Returns the pose of the `idx`-th functional point as `[x, y, z, qw, qx, qy, qz]`

### 3.3 `get_target_point(self, idx: int) -> list[float]`

Returns the pose of the `idx`-th target point as `[x, y, z, qw, qx, qy, qz]`

### 3.4 `get_orientation_point(self, idx: int) -> list[float]`

Returns the pose of the `idx`-th orientation point as `[x, y, z, qw, qx, qy, qz]`

### 3.5 `get_pose(self) -> sapien.Pose`

Returns the global pose of the object in SAPIEN (`.p` is position, `.q` is orientation)

## 4. `ArticulationActor` Class APIs

If the actor was created with method that contains "urdf"(e.g. `create_rand_sapien_urdf_actor`), it will be a subclass of `Actor` called `ArticulationActor`, with the following additional methods:

### 4.1 `get_qlimits(self) -> list[tuple[float, float]]`

Returns a list of joint limits, where each joint limit is a tuple `(min, max)`.

### 4.2 `get_qpos(self) -> list[float]`

Returns the current positions (rotational/positional) of all joints.

### 4.3 `get_qvel(self) -> list[float]`

Returns the current velocities of all joints.


---

## Page: Description Gen (Object & Task)

> Source: [https://robotwin-platform.github.io/doc/usage/description.html](https://robotwin-platform.github.io/doc/usage/description.html)

# Description Gen (Object & Task)

## 1. Object Description

```
# Generate object description for all objects
python3 utils/generate_object_description.py

# Generate object description for a specific type of object with as many objects as this class contains
python3 utils/generate_object_description.py 001_bottle

# Generate object description for a specific object index of a specific type of object
python3 utils/generate_object_description.py 001_bottle --index 0
```

## 2. Task Instruction

```
# Generate 60 task descriptions for a task
python3 utils/generate_task_description.py place_shoe 60
```

It will call for `instruction_num % 12` times of API, each time returning 12 instructions shuffled into 10 seen and 2 unseen instructions.

## 3. Episode Instruction

```
# Generate 60 task descriptions for a task
python3 utils/generate_episode_instructions.py place_shoe franka-panda-D435 1000
```

### 3.1 Parameters:

- `task_name`: Name of the task (JSON file name without extension)
- `setting`: Setting name used to construct the data directory path
- `max_num`: Maximum number of descriptions per episode


---

## Page: Object Annotation

> Source: [https://robotwin-platform.github.io/doc/usage/object-annotation.html](https://robotwin-platform.github.io/doc/usage/object-annotation.html)

# Calibration Tool Instructions

## 1. Rigid Body Object Annotation

### 1.1 Create Calibration Window:

```
python script/create_object_data.py [-s START] model_name

positional arguments:
    model_name            Model name

options:
    -s START, --start START Start id
```

Here, `model_name` is the name of a subdirectory under the `assets/objects/` directory. For example, to calibrate the hammer model located at `assets/objects/020_hammer`, run the command: `python script/create_object_data.py 020_hammer`. A window will then appear as shown below: ![alt text](https://robotwin-platform.github.io/doc/usage/object_marking/image.png)

### 1.2 Calibration Commands:

```
resize:
    Usage:
        resize <x_size> <y_size> <z_size>: Set scaling along x, y, z axes
        resize <size>: Uniformly scale all three axes
    Example:
        resize 0.1
create:
    Usage:
        create <type>: Create (t)arget, (c)ontact, (f)unctional, or (o)rientation point
        create: Waits for input of point name
    Examples:
        create t
        create f
clone:
    Usage:
        clone <type> <id>: Clone a specified type and ID point in place
        clone: Waits for input of point type and ID
    Examples:
        clone t 1: Clones contact point target_1 to create a new target point (e.g., target_2)
rotate:
    Usage:
        rotate <id> <axis> <interval>: Rotate a specified contact point around its own axis by a given interval, generating points belonging to the same group
    Example:
        rotate 1 x 90: Rotates contact_1 around its x-axis every 90 degrees, creating three additional contact points, and writes the group into concat_points_group
align:
    Usage:
        align: Aligns all group points' positions to the first point in the group
remove:
    Usage:
        remove <type> <id>: Removes a point with the specified name
        remove: Waits for input of point name
    Examples:
        remove t 0
save:
    Saves current calibration data — always remember to save!
exit:
    Exits the calibration window
```

As an example using `020_hammer`, entering `create c` creates a cube centered on the object. Use your mouse to select this cube and check "Enable" under the Transform section in the UI window. Then choose "Local" to display the cube's center position and coordinate system, which represents the contact point's location and orientation: ![alt text](https://robotwin-platform.github.io/doc/usage/object_marking/image1.png)

You can move the calibration point's position with the mouse. Click on "Rotate" in the Transform options to adjust the rotation along the x, y, and z axes, changing the point's coordinate system orientation: ![alt text](https://robotwin-platform.github.io/doc/usage/object_marking/image2.png)

Next, add a functional point to the head of the hammer, adjust its orientation, and use the command `create f` to move it to the center position of the hammer head. The adjusted point is shown in the following image: ![alt text](https://robotwin-platform.github.io/doc/usage/object_marking/image3.png)

Finally, enter `save` to save the point information, and then enter `exit` to end the calibration.

Notes :

1. After adjusting the position, you must click "Teleport" under the Transform menu to apply the movement.
2. Always remember to save your changes before exiting the calibration window!

### 1.3 View Calibration Files

Navigate to the asset folder you just calibrated, and you will find a newly generated `model_data{id}.json` file. You can modify the `"scale"` field within this file to adjust the asset's scaling in the simulation environment. ![alt text](https://robotwin-platform.github.io/doc/usage/object_marking/image4.png)

The meanings of each field in the asset can be found in the [model\_data\_info](https://robotwin-platform.github.io/doc/usage/object_marking/model_data_info.html) file.

## 2. URDF Articulation Objects Annotation

### 2.1 Create Calibration Window:

Similar to rigid body object annotation, use the same command to create the articulation calibration window. The calibration program will automatically recognize the asset type.

### 2.2 Calibration Commands:

```
run:
    Usage:
        run
        Press <Ctrl + C> to stop and save information
    Used to obtain stable points through steps, generally selected at the beginning of calibration to determine if running is necessary.
    Since this command does not limit the step upper limit, you need to manually stop running (press Ctrl+C) based on whether the asset in the UI interface is stable.
qpos:
    Usage:
        qpos
    Get the current joint state as the initial pose when loading the asset into the task.
mass:
    Usage:
        mass <m1> <m2> ...: Set the mass of the articulation joint, ensuring that the input matches the displayed link count (excluding base) in order.
    Example:
        mass 0.5 0.05
resize:
    Usage:
        resize <size>: Synchronize the scaling of all three axes of the object
    Example:
        resize 0.1
create:
    Usage:
        create <type> <base_link>: Create (t)arget, (c)ontact, (f)unctional, (o)rientation points
    Example:
        create c link1
rebase:
    Usage:
        rebase <type> <id> <base_link>: Modify the base link of the specified point
    Example:
        rebase c 0 link1
clone:
    Usage:
        clone <type> <id>: Create an in-place copy of the specified point (without base)
    Example:
        clone t 1: Create a new target point (e.g., target_2<link1>) by copying target_1<link1>
rotate:
    Usage:
        rotate <id> <axis> <interval>: Rotate the specified contact point around its own specified axis by the specified interval, generating points belonging to the same group
    Example:
        rotate 1 x 90: Rotate contact_1 around its own x-axis by 90 degrees, generating three contact points, and write the grouping of the four points into concat_points_group
align:
    Usage:
        align: Align the positions of all group points to the first point in the group
remove:
    Usage:
        remove <type> <id>: Remove the specified point (without base)
    Example:
        remove t 0
save:
    Usage:
        save: Save the current calibration data, and make sure to save!
exit:
    Usage:
        exit: Exit the calibration window
```

The calibration process is similar to rigid body object annotation, and you also need to save the data and exit after completion.


---

## Page: Configuring New Embodiment

> Source: [https://robotwin-platform.github.io/doc/usage/new-embodiment.html](https://robotwin-platform.github.io/doc/usage/new-embodiment.html)

# Configure New Embodiment in RoboTwin

> We currently support the Aloha-AgileX, Franka, UR5, Piper, and ARX-X5 robot platforms. For usage instructions, see [Configuation Tutorial](https://robotwin-platform.github.io/doc/usage/configurations.html).

Embodiments are stored in the `assets/embodiments` directory. Each embodiment follows this file structure:

```
# Using Franka as an example
- embodiments
  - franka-panda
    - config.yml # RoboTwin config file
    - curobo_tmp.yml # CuRobo config template
    - collision_franka.yml # CuRobo collision annotations
    - urdf_files/... # URDF files and corresponding GLB, STL files, etc.
```

This guide explains how to configure a new embodiment from scratch, using Franka as an example.

## 1. Step 1: Configure CuRobo Files

For complete configuration instructions, refer to the official documentation: https://curobo.org/tutorials/1\_robot\_configuration.html. This section provides the minimal configuration steps.

### 1.1 Create the embodiment directory and files

```
cd ${ROBOTWIN_ROOT_PATH}
mkdir -p assets/embodiments/new_robot
cd assets/embodiments/new_robot
touch curobo_tmp.yml
touch collision.yml
```

### 1.2 Configure curobo\_tmp.yml

Here's a minimal Franka configuration example:

```
robot_cfg:
  kinematics:
    urdf_path: ${ASSETS_PATH}/assets/embodiments/franka-panda/panda.urdf
    base_link: "panda_link0"
    ee_link: "panda_hand"
    collision_link_names:
      [
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
        "attached_object",
      ]
    collision_spheres: ${ASSETS_PATH}/assets/embodiments/franka-panda/collision_franka.yml
    collision_sphere_buffer: 0.004
    self_collision_ignore: {...}
    self_collision_buffer: {...}
    mesh_link_names: [...]
    lock_joints: {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
    cspace:
      joint_names: ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5", "panda_joint6","panda_joint7","panda_finger_joint1", "panda_finger_joint2"]
      retract_config: [0.2200, -1.4012, -0.0406, -1.4901,  0.3050,  0.4521,  0.2099, 0.04, 0.04]
      null_space_weight: [1,1,1,1,1,1,1,1,1]
      cspace_distance_weight: [1,1,1,1,1,1,1,1,1]
      max_acceleration: 15.0
      max_jerk: 500.0
planner:
  frame_bias: [0., 0., 0.]
```

**Key Parameter Explanations:**

1. **Path Requirements**: Since this is a config template and CuRobo only supports absolute paths, both `urdf_path` and `collision_spheres` must keep the `${ASSETS_PATH}/assets/embodiments/` prefix unchanged. The `${ASSETS_PATH}` variable will be automatically replaced with the absolute path during subsequent operations.
2. **base\_link and ee\_link**: These are the two most important links that directly determine your planning space. Replace these with your robot arm's actual link names.
3. **Collision Configuration**: `collision_link_names` and `collision_spheres` determine self-collision and environment collision detection during planning. For detailed configuration, refer to the "Robot Collision Representation" section at https://curobo.org/tutorials/1\_robot\_configuration.html. All configurations in this repository are based on Isaac Sim 4.2.
4. **Joint Configuration**: `cspace/joint_names` directly determines which joints need planning. This is defined by the URDF and must match the corresponding joint names. The lengths of `retract_config`, `null_space_weight`, and `cspace_distance_weight` must match the length of `joint_names`.
5. **Frame Bias**: For single-arm URDFs, keep `planner/frame_bias` as `[0., 0., 0.]`. For dual-arm setups like ALOHA, slight adjustments are needed (detailed in the [dual-arm configuration section](#dual-arm-urdf-configuration)).

### 1.3 Configure collision.yml

After annotating with Isaac Sim, you'll get collision spheres for different joints. Fill them into collision.yml in this format:

```
collision_spheres:
    panda_link0:
        - "center": [0.0, 0.0, 0.085]
          "radius": 0.03
        # ... more spheres
    panda_link1:
        - "center": [0.0, -0.08, 0.0]
          "radius": 0.035
        # ... more spheres
```

### 1.4 Verify CuRobo Configuration

After configuring CuRobo, verify the setup with a simple forward kinematics test. First, update the `${ASSETS_PATH}`:

```
cd ${ROBOTWIN_ROOT_PATH}
python script/update_embodiment_config_path.py
```

This will generate `curobo.yml` from `curobo_tmp.yml`. Then run this verification code:

```
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

tensor_args = TensorDeviceType()

# Modify to the absolute path of `curobo.yml`
config_file = load_yaml("/abs_path/to/curobo.yml")

urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
kin_model = CudaRobotModel(robot_cfg.kinematics)
q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
out = kin_model.get_state(q)
```

If no errors occur, the configuration is successful.

## 2. Step 2: Configure RoboTwin Config File

### 2.1 Create config.yml

```
cd assets/embodiments/new_robot
touch config.yml
```

### 2.2 Parameter Configuration

Here's a Franka configuration example with detailed explanations:

```
urdf_path: "./panda.urdf"
srdf_path: "./panda.srdf"
joint_stiffness: 1000
joint_damping: 200
gripper_stiffness: 1000
gripper_damping: 200
move_group: ["panda_hand","panda_hand"]
ee_joints: ["panda_hand_joint","panda_hand_joint"]
arm_joints_name: [['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'],
                  ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']]
gripper_name:
  - base: "panda_finger_joint1"
    mimic: [["panda_finger_joint2", 1., 0.]]
  - base: "panda_finger_joint1"
    mimic: [["panda_finger_joint2", 1., 0.]]
gripper_bias: 0.08
gripper_scale: [0.0, 0.04]
homestate: [[0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483],
            [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483]]
delta_matrix: [[0,0,1],[0,-1,0],[1,0,0]]
global_trans_matrix: [[1,0,0],[0,-1,0],[0,0,-1]]
robot_pose: [[0, -0.65, 0.75, 0.707, 0, 0, 0.707],
             [0, -0.65, 0.75, 0.707, 0, 0, 0.707]]
planner: "curobo"
dual_arm: False
rotate_lim: [0.1, 0.8]
grasp_perfect_direction: ['right', 'left']
static_camera_list: 
- name: head_camera
  position: [0.0, 0.8, 0.9]
  forward: [0, -1, 0]
  left: [1, 0, 0]
```

**Parameter Explanations for New Embodiments:**

1. **urdf\_path and srdf\_path**: Relative paths to URDF and SRDF files within `assets/embodiments/new_robot`. These are loaded by Sapien into the simulator and directly determine the physical collision properties.
2. **move\_group**: Used by MPLib, equivalent to CuRobo's `ee_link`. This is a list containing the ee\_links for left and right arms.
3. **ee\_joints**: Since Sapien only supports global pose reading for joints, use the parent joint of the link specified in `move_group`.
4. **arm\_joints\_name**: Joint names, same as CuRobo's `joint_names` parameter, but organized as a 2D list containing joint names for both left and right arms.
5. **gripper\_name**: Controls gripper movement with structure: `list[dict{"base":str, "mimic":[[str, float, float], ...]}, dict{"base":str, "mimic":[[str, float, float], ...]}]`
6. First level list represents left and right grippers
7. Second level dict distinguishes "base" (actively controlled joint) and "mimic" (passive joints)
8. "base": String representing any gripper finger, controlled by `gripper_scale` where `gripper_scale[0]` is closed state and `gripper_scale[1]` is open state
9. "mimic": 2D array where each element contains [str, float1, float2] - joint name, scale, and bias. Joint angle = float1 \* base\_joint + bias
10. **gripper\_bias**: Adjusts distance from `ee_joint` to gripper center. For example, in vertical downward grasping, larger values move the gripper down, smaller values move it up.
11. **homestate**: Initial robot arm state. Set carefully to avoid self-collision that could cause planning failures.
12. **delta\_matrix**: Rotation matrix to unify different ee\_joint coordinate systems. To avoid errors, initially use an identity matrix as placeholder: `[[1,0,0],[0,1,0],[0,0,1]]`.
13. **global\_trans\_matrix**: Rotation matrix to unify ee\_joint pose reading in Sapien. To avoid errors, initially use an identity matrix as placeholder: `[[1,0,0],[0,1,0],[0,0,1]]`.
14. **robot\_pose**: Base\_link placement positions in format `[[x,y,z,qw,qx,qy,qz],[x,y,z,qw,qx,qy,qz]]`. The x-coordinate represents the center position between two arms, recommended as 0. Actual spacing is adjusted in task configs like `demo_randomized.yml`.
15. **dual\_arm**: Boolean indicating whether the URDF is dual-arm (true for ALOHA) or single-arm (false for Franka).
16. **static\_camera\_list**: Adjusts head\_camera position, where `forward` and `left` represent the z-axis and x-axis directions of the camera coordinate system.

## 3. Step 3: Add Embodiment Path

Edit `task_config/_embodiment_config.yml` and add your new robot path:

```
new_robot:
  file_path: "./assets/embodiments/new_robot"
```

**Note**: Your `config.yml` and `curobo_tmp.yml` must be directly located under `file_path`.

## 4. Step 4: Modify Task Config

In your task config (e.g., `task_config/demo_randomized.yml`), change the `embodiment` section to:

```
embodiment:
- new_robot
- new_robot
- 0.8  # Distance between the two robot arms
```

## 5. Step 5: Calibrate delta\_matrix

This calibration requires the desktop environment and is **extremely important**.

### 5.1 Create Temporary URDF

Before calibrating `delta_matrix` and `global_trans_matrix`, you must create a temporary URDF. Using Franka as an example:

```
cd assets/embodiments/franka-panda
cp panda.urdf panda.urdf.save
```

Modify `panda.urdf` by: 1. **Remove or comment out all collision tags** for every link 2. **Remove all joint limits** and change all `revolute` joints to `continuous`

Example modifications:

```
<!-- Comment out collision -->
<link name="panda_link1">
    <visual>
      <geometry>
        <mesh filename="franka_description/meshes/visual/link1.glb"/>
      </geometry>
    </visual> 
    <!-- <collision>
      <geometry>
        <mesh filename="franka_description/meshes/collision/link1.stl"/>
      </geometry>
    </collision> -->
</link>

<!-- Remove joint limits, change revolute to continuous -->
<!-- <joint name="panda_joint3" type="revolute"> -->
<joint name="panda_joint3" type="continuous">
    <origin rpy="1.57079632679 0 0" xyz="0 -0.316 0"/>
    <parent link="panda_link2"/>
    <child link="panda_link3"/>
    <axis xyz="0 0 1"/>
    <!-- Remove this line: <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.1750"/> -->
</joint>
```

### 5.2 Find Valid Pose

The `delta_matrix` unifies coordinate systems across different robot arms. First, run this script to find a valid pose:

```
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

tensor_args = TensorDeviceType()
config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    None,
    num_seeds=20,
    self_collision_check=False,
    self_collision_opt=False,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)
x_values = torch.linspace(0.35, 0.0, 25).tolist() + torch.linspace(0.35, 0.7, 25).tolist()
y_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()
z_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()
quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0')

print("Testing IK solutions for different positions:")
print("x, y, z, success")
for x in x_values:
    for y in y_values:
        for z in z_values:
            goal = Pose(
                position=torch.tensor([[float(x), float(y), float(z)]], device='cuda:0'),
                quaternion=quaternion
            )
            result = ik_solver.solve_single(goal)
            if result.success.item() == True:
                print(f"{x:.2f}, {y:.2f}, {z:.2f}, {result.success}")
```

Expected output:

```
x, y, z, success
0.35, 0.23, 0.09, tensor([[True]], device='cuda:0')
0.35, 0.23, 0.08, tensor([[True]], device='cuda:0')
0.35, 0.23, 0.07, tensor([[True]], device='cuda:0')
...
```

### 5.3 Test in Simulation

Choose any successful xyz coordinates and modify `envs/robot/planner.py` around line 126:

```
## Temporarily add the successful xyz coordinates ##
target_pose_p = [0.35, 0.23, 0.09]  # Example: using 0.35, 0.23, 0.09
target_pose_q = [1., 0., 0., 0.]
## End temporary addition ## 
goal_pose_of_gripper = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
```

Modify `envs/beat_block_hammer.py` to add a temporary test:

```
######## Add temporary test ##########
arm_tag = ArmTag('left')
action = Action(arm_tag, 'move', [-0.05,0.,0.9])
self.move((arm_tag, [action]))
time.sleep(100)
######################################
# Grasp the hammer with the selected arm
self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01, gripper_pos=0.35))
```

Set `render_freq` to a positive number in your task config (e.g., `demo_randomized.yml`), then run:

```
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 5.4 Analyze Coordinate Systems

You should see a visualization similar to this:

![Coordinate System Visualization](https://robotwin-platform.github.io/doc/usage/images/new_embodiment.png)

**Coordinate System Analysis:** - **ee\_joint\_frame**: - **X-axis (red)**: Should point from the link toward the gripper direction - **Y-axis (green)**: Should be parallel to gripper movement direction (positive or negative) - **Z-axis (blue)**: Determined by right-hand rule

- **reference\_frame**:
- **X-axis**: Robot's forward direction
- **Z-axis**: Opposite to gravity direction (upward)
- **Y-axis**: Determined by right-hand rule
- This frame is fixed and consistent across all robots

### 5.5 Calculate delta\_matrix

The `delta_matrix` represents the rotation from ee\_joint frame to reference frame: `{ee_joint}_Rotation_{reference}`.

From the example image above, the delta\_matrix would be:

```
delta_matrix = [[0, 0, 1],
                [0, -1, 0],
                [1, 0, 0]]
```

Update this matrix in your `config.yml`.

## 6. Step 6: Calibrate global\_trans\_matrix

### 6.1 Get Actual Planned Pose

Keep the `time.sleep` in `beat_block_hammer.py` and modify `envs/robot/planner.py` to output the target quaternion:

```
target_pose_p[0] += self.frame_bias[0]
target_pose_p[1] += self.frame_bias[1]
target_pose_p[2] += self.frame_bias[2]
# Remove the hardcoded position and quaternion
# target_pose_p = np.array([0.35, 0.23, 0.09])
# target_pose_q = np.array([1.0, 0.0, 0.0, 0.0])
print('[debug]: target_pose_q: ', target_pose_q)
goal_pose_of_gripper = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
```

Expected output:

```
[debug]: target_pose_q:  [ 1.68244557e-03 -9.98540531e-01 -3.19133105e-04 -5.39803316e-02]
```

**Important**: Use your actual output quaternion values, not the example above. Each robot arm will produce different quaternion values based on its specific configuration.

### 6.2 Test with New Quaternion

Use the output quaternion to test valid positions by modifying the test script, and REMEMBER TO UPDATE THE QUATERNION:

```
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

tensor_args = TensorDeviceType()
config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    None,
    num_seeds=20,
    self_collision_check=False,
    self_collision_opt=False,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)
x_values = torch.linspace(0.35, 0.0, 25).tolist() + torch.linspace(0.35, 0.7, 25).tolist()
y_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()
z_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()

###### REMEMBER TO UPDATE THE QUATERNION ####
#############################################
# Update the quaternion from the debug output
quaternion = torch.tensor([[1.68244557e-03, -9.98540531e-01, -3.19133105e-04, -5.39803316e-02]], device='cuda:0')
#############################################

print("Testing IK solutions for different positions:")
print("x, y, z, success")
for x in x_values:
    for y in y_values:
        for z in z_values:
            goal = Pose(
                position=torch.tensor([[float(x), float(y), float(z)]], device='cuda:0'),
                quaternion=quaternion
            )
            result = ik_solver.solve_single(goal)
            if result.success.item() == True:
                print(f"{x:.2f}, {y:.2f}, {z:.2f}, {result.success}")
```

Expected output:

```
x, y, z, success
0.35, 0.24, 0.27, tensor([[True]], device='cuda:0')
0.35, 0.24, 0.28, tensor([[True]], device='cuda:0')
...
```

### 6.3 Calculate global\_trans\_matrix

Update `envs/robot/planner.py` with a successful position:

```
target_pose_p[0] += self.frame_bias[0]
target_pose_p[1] += self.frame_bias[1]
target_pose_p[2] += self.frame_bias[2]

# Update with successful position, remove debug print and target_pose_q
target_pose_p = np.array([0.35, 0.24, 0.27])
# target_pose_q = np.array([1.0, 0.0, 0.0, 0.0])
# print('[debug]: target_pose_q: ', target_pose_q)
goal_pose_of_gripper = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
```

Replace the entire `play_once(self)` function in `envs/beat_block_hammer.py` and **UPDATE THE DELTA\_MATRIX BELOW**:

```
def play_once(self):
    # Get the position of the block's functional point
    block_pose = self.block.get_functional_point(0, "pose").p
    # Use left arm for testing
    arm_tag = "left"

    arm_tag = ArmTag('left')
    action = Action(arm_tag, 'move', [-0.05,0.,0.9])
    self.move((arm_tag, [action]))

    import transforms3d as t3d
    while True:
        left_ee_global_pose_q = list(self.robot.left_ee.global_pose.q)
        w_R_joint = t3d.quaternions.quat2mat(left_ee_global_pose_q)
        w_R_aloha = t3d.quaternions.quat2mat(action[1][0].target_pose[3:])
        ######## REMEMBER TO UPDATE THE DELTA_MATRIX!!!! ####
        # Update this delta_matrix with your calculated value
        delta_matrix = np.matrix([[0,0,1],[0,-1,0],[1,0,0]])
        #####################################################
        global_trans_matrix = w_R_joint.T @ w_R_aloha @ delta_matrix.T
        print(np.round(global_trans_matrix))
```

Run the simulation again:

```
bash collect_data.sh beat_block_hammer demo_randomized 0
```

Expected output:

```
[[ 1.  0.  0.]
 [ 0. -1.  0.]
 [ 0. -0. -1.]]
```

This is your `global_trans_matrix`. Add it to your `config.yml`.

### 6.4 Clean Up

Restore the modified files:

```
git checkout -- envs/robot/planner.py
git checkout -- envs/beat_block_hammer.py
```

**Congratulations!** Your new embodiment configuration is now complete.

## 7. Dual-Arm URDF Configuration

Dual-arm URDFs have a slightly different structure:

```
# Using ALOHA as an example
- embodiments
  - aloha
    - config.yml # RoboTwin config file
    - curobo_left_tmp.yml # Left arm CuRobo config template
    - curobo_right_tmp.yml # Right arm CuRobo config template
    - collision_aloha_left.yml # Left arm collision annotations
    - collision_aloha_right.yml # Right arm collision annotations
    - urdf_files/... # URDF files and corresponding GLB, STL files, etc.
```

### 7.1 Key Considerations for Dual-Arm Setup:

1. **Frame Bias Configuration**: In `curobo_left_tmp.yml` and `curobo_right_tmp.yml`, if your CuRobo config's `robot_cfg/kinematics/base_link` doesn't match the URDF's `base_link` (e.g., using `fl_base_link` in ALOHA), you need `planner/frame_bias`. This represents the translation vector from the URDF's `base_link` to the CuRobo's `base_link` (e.g., `fl_base_link`). The same applies to the right arm.
2. **Config.yml Settings**: Set `dual_arm: True` in config.yml for dual-arm configurations.

This completes the embodiment configuration process. The setup allows RoboTwin to properly load and control your new robot embodiment in both simulation and planning contexts.


---

## Page: Configuring New Camera

> Source: [https://robotwin-platform.github.io/doc/usage/new-camera.html](https://robotwin-platform.github.io/doc/usage/new-camera.html)

# Configurating New Camera

Modify `task_config/_camera_config.yml` ([Github file](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/task_config/_camera_config.yml)), adding new camera new and configurate `fov`, `h` and `w`, such as:

```
Demo_Camera:
  fovy: 56
  w: 224
  h: 224
```

Finally, modify the camera type in the task config file.

```
camera:
  head_camera_type: Demo_Camera
  wrist_camera_type: D435
```
