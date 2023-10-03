import math 


def create_inner_conn_list(w, h, npc):
    conn_list = []    
    weight = 56
    delay = 1 # 1 [ms]
    nb_col = math.ceil(w/npc)
    nb_row = math.ceil(h/npc)

    pre_idx = -1
    for h_block in range(nb_row):
        for v_block in range(nb_col):
            for row in range(npc):
                for col in range(npc):
                    x = v_block*npc+col
                    y = h_block*npc+row
                    if x<w and y<h:
                        pre_idx += 1                 
                        print(f"{pre_idx} -> ({x},{w+y}) (y:{y})")      
                        # conn_list.append((pre_idx, x, weight, delay))
                        # conn_list.append((pre_idx, w+y, weight, delay))
        
    # save_tuples_to_csv(conn_list, "my_conn_list.csv")
    return conn_list


create_inner_conn_list(8,4,2)