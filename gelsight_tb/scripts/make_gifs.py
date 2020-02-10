import deepdish as dd
import argparse
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np
import ipdb;


def make_gif_from_file(filename, fps=10):
    file = dd.io.load(filename)
    print(f'{len(file)} frames will be added')
    gif_lists = {
        'combined': []
    }
    for x in range(len(file)):
        data = file[x]
        print(data['tb_state'])
        side_by_side = []
        for im in data['images']:
            if im not in gif_lists:
                gif_lists[im] = [data['images'][im]]
            else:
                gif_lists[im].append(data['images'][im])
            side_by_side.append(data['images'][im])
        side_by_side = np.concatenate(side_by_side, axis=1)
        gif_lists['combined'].append(side_by_side)
    for name, list in gif_lists.items():
        clip = ImageSequenceClip(list, fps=fps)
        clip.write_gif(f'{name}-vis.gif', fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='view hdf5 file as python object')
    parser.add_argument('file', action='store')
    args = parser.parse_args()
    make_gif_from_file(args.file)

