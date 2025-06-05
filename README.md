# Installation
To install the dependencies for gsplat, we created an conda environment: 

```bash
conda env create --file gsplat/env2.yml
conda activate two_gsplat
```

Then, we made a new directory called `third_party` under the directory `/home/scur0703/.conda/envs/two_gsplat/lib/python3.11/site-packages/gsplat/cuda/csrc`, and downloaded `glm` as follows: 

```bash
git clone https://github.com/g-truc/glm.git /home/scur0703/.conda/envs/two_gsplat/lib/python3.11/site-packages/gsplat/cuda/csrc/third_party
```

Additionally, to install the required fused-ssim library, we executed the following: 
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5
```
# Resolve pycolmap library issues
When training gsplat, errors has occured about our provided colmap. The errors were related to the installed pycolmap library, specifically in their `scene_manager.py` file. To prevent these errors from occurring, we first find the directory where `scene_manager.py` is stored: 
```bash
find ~/.conda/envs/gsplat_env_test/lib/python3.11/site-packages -name scene_manager.py
```
Once the right directory is found, we used `grep` to find all occurences of `(map`:
```bash
grep -nH "np.array(map" ~/.conda/envs/gsplat_env_fixed/lib/python3.11/site-packages/pycolmap/*.py
```
After that, we provided the directory of all occurences of `np.array(map(` as `file_path` in the file `scene_manager_listmap.py` and the directory of all occurences of `append(map(` as `file_path` in the file `scene_manager_map.py`. We then run the following files in the following order: 
```bash
python scene_manager_listmap.py
python scene_manager_map.py
```

# Room Layouting
We reconstruct indoor room layouts from RGBD video data

# Data - file structure
Place the `depth`, `images` directories containing RGBD imagery data and `views.json` containing camera calibrations in the `res` folder.


