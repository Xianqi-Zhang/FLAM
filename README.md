# <u>FLAM</u>: <u>F</u>oundation Model-Based Body Stabilization for Humanoid <u>L</u>ocomotion <u>a</u>nd <u>M</u>anipulation

<!-- <img src="/assets/flam.png" width=80% height=80%> -->
<img src="/assets/flam.png">


## Installation

* Create a conda environment:
    ```commandline
    conda env create -f environment.yaml
    conda activate humanoid
    ```

* Install [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) benchmark.
    ```commandline
    cd src/env
    pip install -e .
    ```

* Download [data.zip](https://drive.google.com/file/d/1hzV1mFWkYqzNZxbXczXgqUzzAOQnE_ZJ/view?usp=sharing), unzip, and put it in `src/pkgs/RoHM/data`. (You can directly download [data.zip](https://drive.google.com/file/d/1hzV1mFWkYqzNZxbXczXgqUzzAOQnE_ZJ/view?usp=sharing), or download each part from the following links.)

    The directory structure of `/src/pkgs/RoHM/data` is
      - body_models: [SMPL-X body model](https://smpl-x.is.tue.mpg.de/index.html)
      - checkpoints: [RoHM checkpoints](https://github.com/sanweiliti/RoHM/releases/tag/v0)
      - eval_noise_smplx: [Pre-computed motion noise for RoHM](https://drive.google.com/file/d/14H9qglRi9ogPO1dhBUjJWy5_vimDrmCs/view)
      - smplx_vert_segmentation.json: [SMPL-X vertices segmentation](https://meshcapade.wiki/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json)

* **NOTE**:

    - More info can be found in [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) and [RoHM](https://github.com/sanweiliti/RoHM?tab=readme-ov-file). 
    - If you download the environment from [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench), please replace robots.py and wrapper.py, since they have been modified to get camera pose and robot global info, specifically, 
        ```commandline
        class H1 in robots.py
        - camera_pose()
        - global_positioin()
        - global_orientation()

        class ObservationWrapper in wrapper.py
        - observation_space()
        - get_obs() 
        ```
    - The interface of [RoHM](https://github.com/sanweiliti/RoHM?tab=readme-ov-file) has been modified to facilitate calling.

## Training

* Fill in your [WANDB_API_KEY](https://wandb.ai/authorize) in the train.py. (wandb related settings can be found in config_model.py.)
    ```commandline
    os.environ['WANDB_API_KEY'] = 'xxxx'
    ```

* The humanoid task is set in src/config/config_env.py, and can be changed according to choices.
    ```commandline
    group.add_argument('--env', default='h1hand-run-v0', help='e.g. h1hand-walk-v0')
    ```

* Finally, just python script.
    ```commandline
    python train.py
    ```

* **NOTE**:
    - config_env.py: environment related.
    - config_model.py: the proposed method FLAM and the basic policy [TD-MPC2](https://github.com/nicklashansen/tdmpc2) related.
    - config_rohm.py: the foundation model [RoHM](https://github.com/sanweiliti/RoHM) related. Just leave it alone.


## Training Curves
* Training curves can be found in `results/data_vis`.
* The results of baselines are taken from [Humanoid-Bench](https://github.com/carlosferrazza/humanoid-bench) and [CQN-AS](https://github.com/younggyoseo/CQN-AS).
* The results of [CQN-AS](https://github.com/younggyoseo/CQN-AS) on `h1hand-sit_hard-v0`, `h1hand-balance_hard-v0`, `h1hand-stair-v0`, `h1hand-slide-v0`,  and `h1hand-pole-v0`, are reproduced accroding to the official implemention of [CQN-AS](https://github.com/younggyoseo/CQN-AS).
* For visualization, just `python *.py`.


## Possible Problems and Solutions

#### 1. 'GLIBCXX_3.4.30' not found.
```commandline
conda install -c conda-forge libstdcxx-ng
```

#### 2. MESA-LOADER: failed to open swrast/zink.
```commandline
sudo apt install libgl1 libglx-mesa0 libgl1-mesa libgl1-mesa-glx
sudo apt install libgl1-mesa-dev libgl1-mesa-dri
sudo mkdir /usr/lib/dri
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/zink_dri.so /usr/lib/dri/zink_dri.so
```

#### 3. Failed to initialize OpenGL.
- Could not get EGL display.
- Cannot initialize a EGL device display.
- AttributeError: module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'.
```commandline
# * Reference: https://github.com/saqib1707/RL-Robot-Manipulation
conda install -c conda-forge glew
pip install pyrender PyOpenGL-accelerate
pip install pyopengl==3.1.4
```

#### 4. AttributeError: 'mujoco._structs.MjModel' object has no attribute 'cam_orthographic'.
```commandline
pip install dm-control==1.0.19
```

#### 5. 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe'.
```commandline
# * Please uninstall numpy and reinstall.
conda remove --force numpy
pip uninstall numpy
pip install numpy==1.23.4
```

#### 6. AttributeError: 'TensorWrapper' object has no attribute 'seed'.
```commandline
pip install gymnasium==0.29.1
```

#### 7. ERROR: ld.so: object '/usr/lib/x86_64-linux-gnu/libGLEW.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
```commandline
sudo apt install mesa-utils glew-utils libglew-dev
```

#### 8. Fatal error: GL/osmesa.h: No such file or directory
```commandline
sudo apt install libosmesa6-dev
```

* **NOTE**:
  * The code is tested on Ubuntu 20.04 and Linux Mint 22, packages maybe not found in your OS. `apt-cache search package-name` can be used to search relative packages.


## Misc: Code Insights, Lessons, and Suggestions
1. To simplify implementation, the robot's global orientation is replaced with that of the pelvis joint. As a result, when visualizing long trajectory, you may notice a gradual deviation in their global orientation.
2. The foundation model RoHM appears to be sensitive to certain random seeds. In other words, if a result from `[0, 1, 2]` is suboptimal, you can try `10`.
3. TD-MPC2 is an excellent method, but it is extremely slow. Our experiments, including parameter tuning and formal evaluations, took over two months using four RTX 3090 GPUs (and even stronger). You can attempt to speed up the algorithm, but avoid using env parallelism, whether with AsyncVectorEnv or SubprocVecEnv, as it tends to flatten the learning curve. Don't waste time on this, this is the biggest lesson I learned from this work. At least, that was the outcome of my attempts.
4. TODO. (I'll provide updates if I recall anything else.)

## Citation

```commandline
@article{zhang2025flam,
  title={FLAM: Foundation Model-Based Body Stabilization for Humanoid Locomotion and Manipulation},
  author={Zhang, Xianqi and Wei, Hongliang and Wang, Wenrui and Wang, Xingtao and Fan, Xiaopeng and Zhao, Debin},
  journal={arXiv preprint arXiv:2503.22249},
  year={2025}
}
```

## References

This implementation is based on the following repo:
- Humanoid-Bench: https://github.com/carlosferrazza/humanoid-bench
- RoHM: https://github.com/sanweiliti/RoHM
- CQN-AS: https://github.com/younggyoseo/CQN-AS
