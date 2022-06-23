import os
import glob
import shutil

rootdir = "./simulation/"

list = glob.glob(rootdir+"*[0-9]")
print(list)


for p in list:
    gt = glob.glob(p+"/GT*")
    if not os.path.exists(p+"_"):
        os.mkdir(p+"_")
    if len(gt) > 0:
        if not len(glob.glob(p+"_/GT*")) > 0:
            shutil.copy(gt[0], p+"_")
