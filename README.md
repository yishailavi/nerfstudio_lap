## Install
Please make sure you have conda / mamba installed on your system. Please run the following commands to install:

```
git clone https://github.com/yishailavi/nerfstudio_lap.git
cd nerfstudio_lap
mamba create -n nerfstudio_lap python=3.8 -y
mamba activate nerfstudio_lap
pip install -e .
mamba install cuda-toolkit
pip uninstall gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 
ns-install-cli
```

## Running the new method
To train Frequency Aware Gaussian Splatting Decomposition on a scene, please run: 
```
ns-train
splatfacto_lap
--pipeline.model.rasterize_mode
antialiased
--pipeline.model.num_laplacian_levels
3
--pipeline.model.add_level_every
2500
--pipeline.model.num_downscales
0
--pipeline.model.stop_split_at
30000
nerfstudio-data
--data
[PATH]
```