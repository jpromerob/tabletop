


import csv
import argparse
import pdb

'''
This functions initializes a look-up-table that will store mappings from distorted to undistorted pixels
'''
def create_identity_lut(res_x, res_y):
    filepath = f"luts/identity_lut_{res_x}_{res_y}.csv"
    with open (filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=',')
        
        for x in range(res_x): #640
            for y in range(res_y): #480                               
                idx = res_y*x+y
                l = [idx, x, y, -1, -1]
                csv_writer.writerow(l)
     




def parse_args():

    parser = argparse.ArgumentParser(description='Identity LUT generation')

    parser.add_argument('-rx', '--res-x', type=int, help="Resolution X", default=1280)
    parser.add_argument('-ry', '--res-y', type=int, help="Resolution Y", default=720)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()

    create_identity_lut(args.res_x, args.res_y)