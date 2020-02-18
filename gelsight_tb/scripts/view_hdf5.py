import deepdish as dd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import ipdb;


def view_file(filename):
    file = dd.io.load(filename)
    print(len(file))
    x = ''
    while x != 'q':
        x = input('enter which index you want to view info from')
        data = file[int(x)]
        print(data['tb_state'])
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, im in zip(axs, data['images']):
            ax.imshow(data['images'][im].astype(float)/255.)
            ax.set_title(im)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='view hdf5 file as python object')
    parser.add_argument('file', action='store')
    args = parser.parse_args()
    view_file(args.file)

