import argparse
import glob
import pickle as pkl

import numpy as np

parser = argparse.ArgumentParser(description='Generate summary stat pickles from directory containing trajectories')
parser.add_argument('inp_path', metavar='inp_path', type=str, help='directory containing trajectory subdirectories')
parser.add_argument('out_path', metavar='out_path', type=str,
                    help='name of file to to output summary pkl to. If it does not exist, it wil be created.')

args = parser.parse_args()

data_dir = args.inp_path
out_path = args.out_path

traj_paths = glob.glob(data_dir + '/*')
print(traj_paths)

collect_stats = {}

for fname in traj_paths:
    data = pkl.load(open(glob.glob(fname + '/*.pkl')[0], 'rb'))
    for i in range(1, len(data)):
        step_data = data[i]
        for feat in step_data:
            if feat == 'slip':
                continue
            if feat not in collect_stats:
                collect_stats[feat] = []
            collect_stats[feat].append(step_data[feat])

raw_mean = {}
raw_std = {}

for feat in collect_stats:
    if feat == 'slip':
        continue
    collect_stats[feat] = np.array(collect_stats[feat])
    raw_mean[feat] = np.mean(collect_stats[feat])
    raw_std[feat] = np.std(collect_stats[feat])

data = {'mean': raw_mean, 'std': raw_std}
print(data)

with open(out_path, 'wb') as f:
    pkl.dump(data, f)
