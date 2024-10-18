import pickle
import os

def clean_pose_features():
    """ This function removes pickle files in the pose features
        folder that can't be opened
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    directory = f"{current_dir}/../data/features/jaad/poses"
    for root, dir, files in os.walk(directory):
        for filename in files:
            file_path = f"{root}/{filename}"
            with open(file_path, 'rb') as fid:
                try:
                    p = pickle.load(fid)
                except:
                    print(f"Removing {file_path}")
                    os.remove(file_path)

if __name__ == '__main__':
    clean_pose_features()