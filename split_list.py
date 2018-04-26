import os
import subprocess
import sys
import random
import csv
import platform
from utils import mac_remove_file
dir_name="dataset"
# train_folder="Train"
# test_folder="Test"
# clean_data="Mlean_Data"
# ignore_file=".DS_Store"
#
# def remove_ignored_file():
#     if platform.system() == "Darwin":
#         subprocess.call(("find . -name \""+ignore_file+"\"  -delete").split())
def category_to_digit():
    all_classes=getclasses()
    mystruct = dict()
    for i in range(len(all_classes)):
        mystruct[all_classes[i]] = i
    return mystruct
def digit_to_category():
    all_classes=getclasses()
    mystruct = dict()
    for i in range(len(all_classes)):
        mystruct[i] = all_classes[i]
    return mystruct
def getclasses():
    all_classes=sorted(os.listdir(dir_name))
    #all_classes.remove(ignore_file)
    return all_classes

def make_split(trainlength,testlength,version) :
    trainfile="trainlist-"+version+".csv"
    testfile="testlist-"+version+".csv"
    with open(trainfile,"w") as file:
        file.close()
    with open(testfile,"w") as file:
        file.close()
    all_classes=getclasses()
    for i in range(len(all_classes)):
        classname=all_classes[i]
        basepath=os.path.join(dir_name,classname)
        all_audios=os.listdir(basepath)
        num_audio=len(all_audios)
        k = int(num_audio*(trainlength+testlength)/100)
        all_sample = random.sample(all_audios,k)
        train_end = int(num_audio*trainlength/100)
        train_audios=all_sample[:train_end]
        test_audios=all_sample[train_end:]
        with open(trainfile,"a+") as file:
            spamwriter=csv.writer(file)
            for vid in train_audios:
                spamwriter.writerow([os.path.join(basepath,vid),i])
            file.close()
        with open(testfile,"a+") as file:
            spamwriter=csv.writer(file)
            for vid in test_audios:
                spamwriter.writerow([os.path.join(basepath,vid),i])
            file.close()
        # print(all_videos)
        #print(all_sample)
    return trainfile,testfile
    #print(trainlength,testlength,version)
    # global all_classes
    # all_classes=sorted(os.listdir(dir_name))
    # all_classes.remove(ignore_file)


if __name__ == "__main__":

    length=len(sys.argv)
    if length > 1 and sys.argv[1].lower() == "init":
        trainlength = int(sys.argv[2]) if length>2 else 80
        testlength = int(sys.argv[3]) if length>3 else 20
        version = sys.argv[4] if length>4 else "00"
        mac_remove_file()
        make_split(trainlength,testlength,version)
        #print(dir_name)
