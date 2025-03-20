# Frequency-Aware Gaussian Splatting Decomposition
### [Project Page](#) | [Paper](#)
By [Yishai Lavi](#), [Leo Segre](#), and [Shai Avidan](#)

This repository contains the official implementation of **Frequency-Aware Gaussian Splatting Decomposition**, presented in 2025.

![Placeholder screenshot](https://user-images.githubusercontent.com/.../placeholder_screenshot.png)

## Citation
If you find this repository useful, please cite our work:

~~~bibtex
@article{lavi2025frequency,
  title={Frequency-Aware Gaussian Splatting Decomposition},
  author={Yishai Lavi and Leo Segre and Shai Avidan},
  journal={To be published},
  year={2025}
}
~~~

### About  
**Frequency-Aware Gaussian Splatting Decomposition** extends 3D Gaussian Splatting by introducing a **frequency-decomposed framework**, grouping Gaussians into **Laplacian pyramid subbands** to separate low-frequency structures from fine details. This structured frequency control enables advanced 3D editing, dynamic level-of-detail rendering, and improved interpretability for scene manipulation.  

### Installation
This codebase is built on top of [Nerfstudio](https://docs.nerf.studio). To get started, follow these steps:

~~~bash
# 1. Clone the repository
git clone https://github.com/yishailavi/nerfstudio_lap.git
cd nerfstudio_lap

# 2. Create and activate the environment (requires conda/mamba)
mamba create -n nerfstudio_lap python=3.8 -y
mamba activate nerfstudio_lap

# 3. Install the package and dependencies
pip install -e .
mamba install cuda-toolkit

# 4. Install gsplat version 1.4.0
pip uninstall gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0

# 5. Install Nerfstudio CLI tools
ns-install-cli
~~~

### Running
Once installed, you can train **Frequency-Aware Gaussian Splatting Decomposition** on a scene using:

~~~bash
ns-train splatfacto_lap \
  --pipeline.model.rasterize_mode antialiased \
  --pipeline.model.num_laplacian_levels 3 \
  --pipeline.model.add_level_every 2500 \
  --pipeline.model.num_downscales 0 \
  --pipeline.model.stop_split_at 30000 \
  nerfstudio-data --data [PATH]
~~~

Replace `[PATH]` with the path to your dataset.

### Built On
<a href="https://github.com/nerfstudio-project/nerfstudio">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/_images/logo.png" />
    <img alt="nerfstudio logo" src="https://docs.nerf.studio/_images/logo.png" width="150px" />
</picture>
</a>

- A collaboration friendly studio for NeRFs

