#!/bin/bash
wget http://campar.in.tum.de/files/graphto3d/final_checkpoints.zip
wget http://campar.in.tum.de/files/graphto3d/atlasnet/model_70.pth
unzip final_checkpoints.zip
mkdir atlasnet
mv model_70.pth ./atlasnet/
