import json
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

# TODO: (felix) Deal with label data when video data changes!

class PoseDataset(Dataset):
    """PoseDataset takes a path as input argument and searches for subfolders
    within this path. It will list all json files within the subfolders and
    return x,y data in as numpy matrices of shape (25, seq_len*fps).

    pose_data_path = 'some/path/to/data'
    label_data_path = 'some/path/to/labels'
    target = 'fall'
    my_dataset = PoseDataset(pose_data_path, label_data_path, target, seq_len=1, fps=25)
    index = 0
    x, y = my_dataset.__getitem__(index)
    """
    def __init__(self, pose_data_path, label_data_path, target, seq_len=1, fps=25, debug=False):
        # take input arguments
        self.seq_len = seq_len
        self.fps = fps
        self.seq_n = int(np.round(seq_len * fps))
        self.label_path = label_data_path
        self.target = target
        self.label_file = ''
        self.label_data = []
        self.debug = debug

        # scan given path for subfolders
        pose_data_folders = list()
        if (debug):
            print(f'Scanning file tree...')
        for folder in os.listdir(pose_data_path):
            full_path = os.path.join(pose_data_path, folder)
            if os.path.isdir(full_path):
                if (debug):
                    print(f'\t- {folder}')
                pose_data_folders.append(full_path)

        # list all json files/frames including path
        self.files = list()
        for folder in pose_data_folders:
            if (debug):
                print(folder)
            tmp = os.listdir(folder)
            for file in tmp:
                if file.endswith('.json'):
                    self.files.append(os.path.join(folder, file))
            if (debug):
                print(len(self.files), len(tmp))

    def __len__(self):
        # return number of files/frames with pose data
        return int(np.ceil(len(self.files)/self.seq_n))

    def __getitem__(self, index):
        # 75 is the length of openpose keypoint 2d output (x/y/confidence)
        data_x = np.zeros((25, self.seq_n))
        data_y = np.zeros((25, self.seq_n))
        #
        for kk, point in enumerate(self.files[index * self.seq_n : (index+1) * self.seq_n]):
            next_label_file = os.path.join(
                self.label_path,
                os.path.dirname(point).split(os.path.sep)[-1] + '.csv'
            )
            if (next_label_file != self.label_file):
                self.label_file = next_label_file
                if (self.debug):
                    print(next_label_file)
                # load new label file
                self.label_data = pd.read_csv(next_label_file)
            label = self.label_data[index * self.seq_n : (index+1) * self.seq_n][self.target].values
            with open(point) as f:
                pose_data = json.loads(f.read())['people']
                if len(pose_data) > 0:
                    data = np.array(pose_data[0]['pose_keypoints_2d'])
                    data_x[:, kk] = data[::3]  # x data
                    data_y[:, kk] = data[1::3]  # y data
                    # data_conf[:, kk] = data[2::3]  # confidence
        return data_x, data_y, label
