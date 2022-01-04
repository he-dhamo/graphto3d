#!/bin/bash
wget http://campar.in.tum.de/files/3RScan/rescans.txt
wget http://campar.in.tum.de/files/3RScan/train_ref.txt
wget http://campar.in.tum.de/files/3RScan/val_ref.txt
wget http://campar.in.tum.de/files/3RScan/train_scans.txt
wget http://campar.in.tum.de/files/3RScan/val_scans.txt
wget http://campar.in.tum.de/files/3RScan/3RScan.json
cat ./train_ref.txt > references.txt
echo >> references.txt
cat ./val_ref.txt >> references.txt
rm train_ref.txt
rm val_ref.txt
