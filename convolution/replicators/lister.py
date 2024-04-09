
import numpy as np
import pdb
import math

SUB_WIDTH = 8
SUB_HEIGHT = 4
WIDTH = 20 # dim.fl
HEIGHT = 10 # dim.fw

def create_array(width, height, offset):
    # Create meshgrid of x and y coordinates
    y, x = np.meshgrid(range(height), range(width))
    # Calculate the values based on the equation val = x + y*N
    val = x + y * width + offset
    return val

def create_index_matrix():
    
    block_width = SUB_WIDTH
    block_height = SUB_HEIGHT

    matrix = np.zeros((WIDTH, HEIGHT))
    nb_h_blocks = math.ceil(WIDTH/SUB_WIDTH)
    nb_v_blocks = math.ceil(HEIGHT/SUB_HEIGHT)

    print(f"Blocks: {nb_h_blocks}x{nb_v_blocks}]")

    offset = 0
    for v_block in range(nb_h_blocks):
        for h_block in range(nb_h_blocks):

            if (h_block+1)*block_width <= WIDTH:
                sub_block_width = block_width
            else:
                sub_block_width = WIDTH-h_block*block_width

            if (v_block+1)*block_height <= HEIGHT:
                sub_block_height = block_height
            else:
                sub_block_height = HEIGHT-v_block*block_height
            print(f"Block size: {sub_block_width}x{sub_block_height}]")
        
            sub_block = create_array(sub_block_width, sub_block_height, offset)
            offset = offset + sub_block_width*sub_block_height

            x_min = h_block*block_width
            x_max = x_min + sub_block_width
            y_min = v_block*block_height
            y_max = y_min + sub_block_height
            matrix[x_min:x_max, y_min:y_max] = sub_block
            print(sub_block)


    return matrix


    pdb.set_trace()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            print(f"{matrix[x,y]}")

if __name__ == "__main__":

    create_index_matrix()