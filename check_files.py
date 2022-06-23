import os
import glob
import shutil

rootdir = "./simulation/"

list = glob.glob(rootdir+"*[0-9]")

for p in list:
    gt = glob.glob(p+"/SoS*")
    if len(gt) != 8:
        print(p)
    gt = glob.glob(p+"/GT*")
    if len(gt) != 1:
        print(p)

list = glob.glob(rootdir+"*[0-9]_")

for p in list:
    gt = glob.glob(p+"/SoS*")
    if len(gt) != 8:
        print(p)
    gt = glob.glob(p+"/GT*")
    if len(gt) != 1:
        print(p)