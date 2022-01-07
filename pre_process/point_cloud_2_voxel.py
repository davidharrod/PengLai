from typing import Type
from pyntcloud import PyntCloud
import numpy as np
from pyntcloud.io import obj
from . import binvox_rw
import os
import shutil

PLY = "PLY"
NPY = "NPY"
BINVOX = "BINVOX"
FILE = "FILE"
FILELIST = "FILELIST"
OBJ = "OBJ"


def _get_filename(file):
    return os.path.splitext(os.path.basename(file))[0]


def _get_suffix(filename):
    """Get the capital suffix of filename"""
    return (os.path.splitext(filename)[-1][1:]).upper()


def _is_in_file_list(file, filename_list):
    filename = _get_filename(file)
    return 1 if filename in filename_list else 0


def _read_file(file, file_type):
    if file_type == PLY or file_type == NPY:
        return PyntCloud.from_file(file)
    elif file_type == BINVOX:
        with open(file, "rb") as binvox_file:
            return binvox_rw.read_as_3d_array(binvox_file)
    else:
        raise AttributeError(f"{file_type} format is not supported currently.")


def _read_files(file_list, file_type):
    """Open files."""
    model_list = []
    for file in file_list:
        model_list.append(_read_file(file, file_type))
    return model_list


def _filter_file(file):
    num_vertices = 0
    num_faces = 0
    with open(file) as file:
        for line in file.readlines():
            tokens = line.split()
            line_type = tokens[0]
            if not tokens:
                continue
            if line_type == "v":
                num_vertices += 1
            elif line_type == "f":
                num_faces += 1
    return 0 if num_vertices > 800 or num_faces > 2800 else 1


def _filter_files(file_list):
    filtered_file_list = []
    for i, file in enumerate(file_list):
        if _get_suffix(file) == OBJ:
            if _filter_file(file):
                filtered_file_list.append(file)
        else:
            raise TypeError("This function is for .obj files only.")
    return filtered_file_list


def _get_file_list(path, file_type=None):
    file_list = []
    if not os.path.exists:
        raise IOError(f"{path} does not exist.")
    else:
        if not os.path.isdir(path):
            raise IOError(f"{path} is not a valid directory.")
        else:
            for _, _, files in os.walk(path):
                for file in files:
                    if file_type:
                        if file.endswith(file_type.lower()):
                            file_list.append(os.path.join(path, file))
                    else:
                        file_list.append(os.path.join(path, file))
                if file_list:
                    return file_list
                else:
                    raise IOError(
                        f"There is no valid {file_type} file in {path}.")


def _get_matched_files(file_list_1, file_list_2):
    """Get matched files from path2"""
    filename_list = list(map(_get_filename, file_list_1))
    matched_file_list = []
    for file in file_list_2:
        flag = _is_in_file_list(file, filename_list)
        if flag:
            matched_file_list.append(file)
    return matched_file_list


def _try_2_create_directory(target_dir, dir_name):
    if not target_dir:
        raise ValueError(
            "Please set the target directory before using this funciton.")
    try:
        os.makedirs(os.path.join(target_dir, dir_name))
    except FileExistsError:
        return os.path.join(target_dir, dir_name)
    else:
        return os.path.join(target_dir, dir_name)


def _copy_file(file_list, target_file_path):
    for file in file_list:
        try:
            shutil.copy(file, target_file_path)
        except TypeError:
            raise TypeError(f"Directory {target_file_path} is not valid.")
        except FileNotFoundError:
            continue
    return None


def pre_process(file_path_1, file_path_2):
    """Preprocess polygon mesh and binvox dataset."""
    file_list_1 = _get_file_list(file_path_1)
    file_list_2 = _get_file_list(file_path_2)
    # Filter.
    file_list_1 = _filter_files(file_list_1)
    # Get matched files.
    file_list_2 = _get_matched_files(file_list_1, file_list_2)
    return file_list_1, file_list_2


def subdivide(file_path,
              target_file_path,
              train_size,
              val_size,
              train_dir_name="train",
              val_dir_name="val",
              test_dir_name="test"):
    """Subdivide dataset according to percentile."""
    file_list = _get_file_list(file_path)
    num_files = len(file_list)
    train_file_path = _try_2_create_directory(target_file_path, train_dir_name)
    val_file_path = _try_2_create_directory(target_file_path, val_dir_name)
    test_file_path = _try_2_create_directory(target_file_path, test_dir_name)
    _copy_file(file_list[0:int(train_size * num_files)], train_file_path)
    _copy_file(
        file_list[int(train_size * num_files):int(train_size * num_files) +
                  int(val_size * num_files)], val_file_path)
    _copy_file(
        file_list[int(train_size * num_files) +
                  int(val_size * num_files):num_files], test_file_path)
    print("Subdivide successfully!")
    return None


def _get_binarized_vector_point_cloud(point_cloud, voxel_size=0.1):
    voxel_grid_id = point_cloud.add_structure("voxelgrid",
                                              size_x=voxel_size,
                                              size_y=voxel_size,
                                              size_z=voxel_size)
    voxel_grid = point_cloud.structures[voxel_grid_id]
    return voxel_grid.get_feature_vector(mode="binary")


def _get_binarized_vector_binvox(voxel):
    return voxel.data * 1


def get_binarized_voxel_vector(path, file_type, mode=FILE):
    """Take in files' path and return a list of binarized voxel vectors accordingly."""
    if mode == FILELIST:
        file_list = _get_file_list(path, file_type)
    elif mode == FILE:
        file_list = [path]
    else:
        raise AttributeError(
            f"{mode} is not supported currently. Try FILE or FILELIST instead."
        )
    model_list = _read_files(file_list, file_type)
    if file_type == PLY or file_type == NPY:
        binarized_vector = list(
            map(_get_binarized_vector_point_cloud, model_list))
    elif file_type == BINVOX:
        binarized_vector = list(map(_get_binarized_vector_binvox, model_list))
    else:
        raise AttributeError(f"{file_type} format is not supported currently.")
    binarized_vector = np.asarray(binarized_vector)
    dim = binarized_vector.shape[1]
    return binarized_vector.reshape(dim, dim, dim)
