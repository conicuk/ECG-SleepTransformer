import numpy as np

def create_r_permute_shhs(num_files):
    indices = np.arange(num_files)
    np.save('E:/coding/sleep_stage/utils/r_permute_shhs2.npy', indices)


num_files = 100
create_r_permute_shhs(num_files)