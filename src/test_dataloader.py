from pose_data_loader import PoseDataset


if __name__ == '__main__':
        # minimum parameters
        pose_data_path = '../data/pose/'
        label_data_path = '../data/label/'
        target = 'fall'

        # optional parameters
        sequence_length_s = 0.5
        fps = 25
        # the batch_size is determined by round(sequence_length * fps)
        # missing frames are filled with 0's

        dataset = PoseDataset(pose_data_path, label_data_path, target,
                              seq_len=sequence_length_s, fps=fps)

        # get first batch from data loader
        x, y, l = dataset.__getitem__(0)
        print(f'x_pos = {x}\ny_pos = {y}\nlabel = {l}')
