# Graph-to-3D

This is the official implementation of the paper **Graph-to-3d: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs | <a href="https://arxiv.org/pdf/2108.08841.pdf">arxiv</a>** <br/>
Helisa Dhamo*, Fabian Manhardt*, Nassir Navab, Federico Tombari<br/>
**ICCV 2021**

We address the novel problem of fully-learned 3D scene generation and manipulation from scene graphs, in which a user can specify
in the nodes or edges of a semantic graph what they wish to see in the 3D scene.

If you find this code useful in your research, please cite
```
@inproceedings{graph2scene2021,
  title={Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes using Scene Graphs},
  author={Dhamo, Helisa and Manhardt, Fabian and Navab, Nassir and Tombari, Federico},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Setup

We have tested it on Ubuntu 16.04 with Python 3.7 and PyTorch 1.2.0

### Code
```bash
# clone this repository and move there
git clone https://github.com/he-dhamo/graphto3d.git
cd graphto3d
# create a conda environment and install the requirments
conda create --name g2s_env python=3.7 --file requirements.txt 
conda activate g2s_env          # activate virtual environment
# install pytorch and cuda version as tested in our work
conda install pytorch==1.2.0 cudatoolkit=10.0 -c pytorch
# more pip installations
pip install tensorboardx graphviz plyfile open3d==0.9.0.0 open3d-python==0.7.0.0 
# Set python path to current project
export PYTHONPATH="$PWD"
```

To evaluate shape diversity, you will need to setup the Chamfer distance. Download the extension folder from the
<a href="https://github.com/TheoDEPRELLE/AtlasNetV2">AtlasNetv2 repo</a> and install it following their instructions:
```
cd ./extension
python setup.py install
```

To download our checkpoints for our trained models and the Atlasnet weights used to obtain shape features:
```
cd ./experiments
chmod +x ./download_checkpoints.sh && ./download_checkpoints.sh
```

### Dataset

Download the <a href="https://waldjohannau.github.io/RIO/#download">3RScan dataset</a> from their official site. You will need to download
the following files using their script:
```
python download.py -o /path/to/3RScan/ --type semseg.v2.json
python download.py -o /path/to/3RScan/ --type labels.instances.annotated.v2.ply
```
Additionally, download the metadata for 3RScan:
```
cd ./GT
chmod +x ./download_metadata_3rscan.sh && ./download_metadata_3rscan.sh
```
Download the <a href="https://3dssg.github.io/#download" >3DSSG data</a> files to the `./GT` folder:
```
chmod +x ./download_3dssg.sh && ./download_3dssg.sh
```

We use the scene splits with up to 9 objects per scene from the 3DSSG paper.
The relationships here are preprocessed to avoid the two-sided annotation for spatial relationships, as these can lead 
to paradoxes in the manipulation task. Finally, you will need our directed aligned 3D bounding boxes introduced in our
<a href="https://he-dhamo.github.io/Graphto3D/#download">project page</a>. The following scripts downloads these data.
```
chmod +x ./download_postproc_3dssg.sh && ./download_postproc_3dssg.sh
```

Run the `transform_ply.py` script from **this repo** to obtain 3RScan scans in the correct alignment:
```
cd ..
python scripts/transform_ply.py --data_path /path/to/3RScan
```

## Training

To train our main model with shared shape and layout embedding run:

```
python scripts/train_vaegan.py --network_type shared --exp ./experiments/shared_model --dataset_3RScan ../3RScan_v2/data/ --path2atlas ./experiments/atlasnet/model_70.pth --residual True
```

To run the variant with separate (disentangled) layout and shape features:
```
python scripts/train_vaegan.py --network_type dis --exp ./experiments/separate_baseline --dataset_3RScan ../3RScan_v2/data/ --path2atlas ./experiments/atlasnet/model_70.pth --residual True
```

For the 3D-SLN baseline run:
```
python scripts/train_vaegan.py --network_type sln --exp ./experiments/sln_baseline --dataset_3RScan ../3RScan_v2/data/ --path2atlas ./experiments/atlasnet/model_70.pth --residual False --with_manipulator False --with_changes False --weight_D_box 0 --with_shape_disc False
```



One relevant parameter is `--with_feats`. If set to true, this tries to read shape features directly instead of reading 
point clouds and feading them in AtlasNet to obtain the feature. If features are not yet to be found, it generates them 
during the first epoch, and reads these stored features instead of points in the next epochs. This saves a lot of time 
at training. 

Each training experiment generates an `args.json` configuration file that can be used to read the right parameters during evaluation.

## Evaluation

To evaluate the models run
```
python scripts/evaluate_vaegan.py --dataset_3RScan ../3RScan_v2/data/ --exp ./experiments/final_checkpoints/shared --with_points False --with_feats True --epoch 100 --path2atlas ./experiments/atlasnet/model_70.pth --evaluate_diversity False
```
Set `--evaluate_diversity` to `True` if you want to compute diversity. This takes a while, so it's disabled by default.
To run the 3D-SLN baseline, or the variant with separate layout and shape features, simply provide the right experiment folder in `--exp`.

## Acknowledgements

This repository contains code parts that are based on 3D-SLN and AtlasNet. We thank the authors for making their code available.
