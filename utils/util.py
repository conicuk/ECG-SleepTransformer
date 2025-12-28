import os
import numpy as np
from glob import glob

def load_data_shhs(np_data_path, task):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    if task=="shhs1":
        r_p_path = "E:/coding/sleep_stage/utils/r_permute_shhs1.npy"
    elif task=="shhs2":
        r_p_path = "E:/coding/sleep_stage/utils/r_permute_shhs2.npy"
    print(f"파일을 불러오는 경로: {r_p_path}")

    r_permute = np.load(r_p_path)
    print(r_permute)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    return npzfiles

def load_data_mesa(np_data_path):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = "E:/coding/sleep_stage/utils/r_permute_mesa.npy"
    print(f"파일을 불러오는 경로: {r_p_path}")

    r_permute = np.load(r_p_path)
    print(r_permute)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    return npzfiles

def load_data_isruc_s1(np_data_path):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = "E:/coding/sleep_stage/utils/r_permute_isruc_s1.npy"
    print(f"파일을 불러오는 경로: {r_p_path}")

    r_permute = np.load(r_p_path)
    print(r_permute)
    npzfiles = np.asarray(files, dtype='<U200')[r_permute]
    return npzfiles
