# Installation

SafeDrive builds on the NAVSIM devkit, so install the
[NAVSIM environment](https://github.com/autonomousvision/navsim?tab=readme-ov-file#getting-started-)
first, then add the packages below.

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .

# torch and mmcv have to agree: mmcv 2.1.0 is built against torch 2.1
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install openmim
mim install mmcv==2.1.0
pip install einops lmdb
```

A mismatch between the installed torch and the torch mmcv was compiled against
shows up as an `mmcv._ext` ABI error on the first import, not at install time.

## Dataset

Download the OpenScene / navsim splits with the scripts in
[`download/`](../download). [`super_download.sh`](../download/super_download.sh)
parallelises the downloads through tmux.

```bash
cd download
bash super_download.sh
```

Then point the usual NAVSIM environment variables at the result:

```bash
export OPENSCENE_DATA_ROOT=/path/to/dataset
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/exp
```

The feature cache is read through the relative path
`dataset/sensor_blobs/{trainval,test}`, so keep a `dataset` symlink at the
repository root pointing at `OPENSCENE_DATA_ROOT`.
