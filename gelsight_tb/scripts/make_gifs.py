import deepdish as dd
import argparse
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np
import ipdb;
import cv2
import numpy as np


def make_gif_from_file(filename, fps=2):
    file = dd.io.load(filename)
    print(f'{len(file)} frames will be added')
    gif_lists = {
        'combined': []
    }
    for x in range(len(file)):
        data = file[x]
        print(data['tb_state'])
        side_by_side = []
        images = data['raw_images']
        for im in images:
            if im == 'external':
                images[im] = np.rot90(images[im], k=2)
            if im not in gif_lists:
                gif_lists[im] = [images[im]]
            else:
                gif_lists[im].append(images[im])
            side_by_side.append(images[im])
        gif_lists['combined'].append(conpadenate(side_by_side))
    for name, list in gif_lists.items():
        clip = ImageSequenceClip(list, fps=fps)
        clip.write_gif(f'{name}-vis.gif', fps=fps)


def conpadenate(images):
    shape = images[0].shape
    for i, image in enumerate(images):
        if image.shape != shape:
            images[i] = cv2.resize(image, dsize=shape[:2][::-1])
    return np.concatenate(images, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make gif')
    parser.add_argument('file', action='store')
    args = parser.parse_args()
    make_gif_from_file(args.file)

