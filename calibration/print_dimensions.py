
import sys 
sys.path.append('../common')
from tools import Dimensions, get_shapes

dim = Dimensions.load_from_file('../common/homdim.pkl')

print("\nFull - Scaled Pixel's Space:")
print(f"\tFrom {int(dim.l)}x{int(dim.w)} to {int(dim.l*dim.hs)}x{int(dim.w*dim.hs)}")
print(f"\tHomography Scale: {dim.hs} (dim.hs)")


print("\nTable - Scaled Pixel's Space:")
print(f"\tTable Length: {dim.fl} (dim.fl)")
print(f"\tTable Height: {dim.fw} (dim.fw)") 
print(f"\tInter-LED Length: {dim.il} (dim.il)")
print(f"\tInter-LED Height: {dim.iw} (dim.iw)")
print(f"\tGoal size: {dim.gs} (dim.gs)")
print(f"\tPuck Radius: {dim.pr} (dim.pr)")
