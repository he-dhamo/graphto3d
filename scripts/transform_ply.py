import numpy as np
from plyfile import PlyData
import os
import argparse
import json
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../3RScan_v2/data')
parser.add_argument('--scan3r_json', type=str, default='./GT/3RScan.json', help="3Rscan")
parser.add_argument('--references_file', type=str, default='./GT/references.txt', help="references")
parser.add_argument('--rescans_file', type=str, default='./GT/rescans.txt', help="rescans")
parser.add_argument('--scan_id', type=str, default='', help="scan_id, apply for all scans if empty")
parser.add_argument('--filename_in', type=str, default='labels.instances.annotated.v2.ply', help="input file name")
parser.add_argument('--filename_out', type=str, default='labels.instances.align.annotated.ply', help="transformed file name")
opt = parser.parse_args()


def resave_ply(filename_in, filename_out, matrix):
    """ Reads PLY file from filename_in, transforms with matrix and saves a PLY file to filename_out.
    """
    with open(filename_in, 'rb') as file:
        plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()
    # shape = np.array(points.shape)
    points4f = np.insert(points, 3, values=1, axis=1)
    points = points4f * matrix
    plydata['vertex']['x'] = np.asarray(points[:,0]).flatten()
    plydata['vertex']['y'] = np.asarray(points[:,1]).flatten()
    plydata['vertex']['z'] = np.asarray(points[:,2]).flatten()
    # make sure its binary little endian
    plydata.text = False
    plydata.byte_order = '<'
    # save
    plydata.write(filename_out)


def read_transform_matrix():
    rescan2ref = {}
    with open(opt.scan3r_json, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(scans["transform"]).reshape(4,4)
    return rescan2ref


def main():
    rescan2ref = read_transform_matrix()
    counter = 0
    if os.path.exists(opt.rescans_file):
        with open(opt.rescans_file, 'r') as f:
            for line in f:
                print(counter, "/ 1004 rescans ", line.rstrip())
                scan_id = line.rstrip()
                if (opt.scan_id != "") and (scan_id != opt.scan_id):
                    continue
                file_in = os.path.join(opt.data_path, scan_id, opt.filename_in)
                file_out = os.path.join(opt.data_path, scan_id, opt.filename_out)
                if scan_id in rescan2ref: # if not we have a hidden test scan
                    resave_ply(file_in, file_out, rescan2ref[scan_id])
                counter += 1
    counter = 0
    if os.path.exists(opt.references_file):
        with open(opt.references_file, 'r') as f:
            for line in f:
                print(counter, "/ 432 reference ", line.rstrip())
                scan_id = line.rstrip()
                if (opt.scan_id != "") and (scan_id != opt.scan_id):
                    continue
                file_in = os.path.join(opt.data_path, scan_id, opt.filename_in)
                file_out = os.path.join(opt.data_path, scan_id, opt.filename_out)
                copyfile(file_in, file_out)
                counter += 1


if __name__ == "__main__": main()
