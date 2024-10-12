import yaml
import glob
import pickle
import os


def get_file_path(file_path):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, file_path)
    return file_path


def load_yaml(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    print(file_path)
    assert(len(files) == 1)
    cfg = yaml.safe_load(open(files[0], 'r'))
    return cfg


def load_pickle(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    assert(len(files) == 1)
    data = pickle.load(open(files[0], 'rb'))
    return data
