import os
import argparse
import pdb


def parse_args():

    parser = argparse.ArgumentParser(description='Master of C puppet')

    parser.add_argument('-sn', '--simname', type=str, help="Simulation Name", default="None")
    parser.add_argument('-md', '--mode', type=str, help="save|show", default="")
    parser.add_argument('-st', '--show-target', action='store_true', help="Show Target")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args.show_target:
        target = 1
    else:
        target = 0

    if args.mode == "save":
        simname = args.simname
        os.system("rm *.*video.dat")
        os.system(f"./c_code/recorder.exe video algebraic 4 1")
        os.system(f"mv raw_video.dat backup/{simname}_raw_video.dat")
        os.system(f"mv cnn_video.dat backup/{simname}_cnn_video.dat")
    elif args.mode == "show":
        simname = args.simname.replace("backup/", "").replace("_video.dat","").replace("_raw","").replace("_cnn","")
        os.system(f"cp backup/{simname}_raw_video.dat raw_video.dat")
        os.system(f"cp backup/{simname}_cnn_video.dat cnn_video.dat")
        os.system(f"./c_code/vidreader.exe 4 {target} 4")
        os.system("rm *.*video.dat")
    else:
        print("Wrong mode")
        quit()
