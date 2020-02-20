import deepdish as dd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import attrdict
from gelsight_tb.utils.obs_to_np import obs_to_action
from moviepy.editor import ImageSequenceClip


def plot_images(data, raw=True):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    if raw:
        images = data['raw_images']
    else:
        images = data['images']
    for ax, im in zip(axs, images):
        ax.imshow(images[im].astype(float) / 255.)
        ax.set_title(im)
    return fig, axs


def make_gif_of_traj(filename, output_filename, raw=True, fps=5):
    file = dd.io.load(filename)
    traj_len = len(file)
    print(f'trajectory length is {traj_len}')
    traj_images = []
    a = attrdict.AttrDict({'mean': 0, 'scale': 1})
    for i in range(1, traj_len):
        data = file[i]
        act = obs_to_action(data, file[i-1], a)
        sorted_act = sorted(act, key=np.abs)
        if np.abs(sorted_act[0]) == 0 and sorted_act[1] == 0:
            print('alert alert')
        print(act)
        #fig, axs = plot_images(data, raw=raw)
        #fig.canvas.draw()
        #image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #traj_images.append(image)
    #clip = ImageSequenceClip(traj_images, fps=fps)
    print(f'Writing out to {output_filename}.mp4')
    #clip.write_videofile(f'{output_filename}.mp4', fps=fps)
    print(f'Writing out to {output_filename}.gif')
    #clip.write_gif(f'{output_filename}.gif', fps=fps, program='imageio', opt='wu')


def view_file_interactive(filename, raw=True):
    file = dd.io.load(filename)
    print(len(file))
    x = ''
    while x != 'q':
        x = input('enter which index you want to view info from')
        try:
            ind = int(x)
            data = file[ind]
            print(data['tb_state'])
            fig, axs = plot_images(data, raw=raw)
            plt.show()
        except ValueError:
            print('not a valid index?')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='view hdf5 file as python object')
    parser.add_argument('file', action='store')
    parser.add_argument('--make_traj', action='store_true', dest='m_traj')
    parser.add_argument('--output_name ', action='store', type=str, dest='output_dest')

    args = parser.parse_args()
    if args.m_traj:
        make_gif_of_traj(args.file, args.output_dest)
    else:
        view_file_interactive(args.file)

