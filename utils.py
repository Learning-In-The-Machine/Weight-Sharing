import os

def build_directory(path, current_path=''):
    # iterate through folders in specifide path
    for folder in path.split('/'):
        current_path += folder +'/'
        # if it doesn't exist build that director
        if not os.path.exists(current_path):
            os.mkdir(current_path)
