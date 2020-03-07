import numpy as np


def normalize(arr, mean, std):
    return 1.0 * (arr - np.array(mean)) / np.array(std)


def denormalize(arr, mean, std):
    return (arr * np.array(std)) + np.array(mean)


def obs_to_state(obs, norm_conf, should_normalize=True):
    if norm_conf is None or 'mean' not in norm_conf:
        norm_conf = {'mean': np.zeros(4), 'std': np.ones(4)}
    x = obs['tb_state']['x']
    y = obs['tb_state']['y']
    z = obs['tb_state']['z']
    dynamixel = obs['dynamixel_state']

    unnormalized_state = np.concatenate((np.array([x, y, z]), np.array([dynamixel])))
    if not should_normalize:
        return unnormalized_state
    normalized_state = normalize(unnormalized_state, norm_conf['mean'], norm_conf['std'])
    return normalized_state


def obs_to_opto(obs, norm_conf, should_normalize=True):
    if norm_conf is None or 'mean' not in norm_conf:
        norm_conf = {'mean': np.zeros(1), 'std': np.ones(1)}
    forces = np.array(obs['optoforce'])
    if should_normalize:
        forces = normalize(forces, norm_conf['mean'], norm_conf['std'])
    return np.array(forces).astype(np.float32)


def obs_to_images(obs):
    images_dict = obs['images']
    images = []
    for key in sorted(images_dict.keys()):
        if key == 'external':
            images.append(np.rot90(images_dict[key], k=2))
        else:
            images.append(images_dict[key])
    return [image.astype(np.uint8) for image in images]


def visualize_images(obs):
    images_dict = obs['images']
    import matplotlib.pyplot as plt            
    for key in images_dict.keys():
        img = images_dict[key]
        plt.imshow(img)
        plt.show()


def obs_to_action(obs_1, obs_2, norm_conf):
    if norm_conf is None or 'mean' not in norm_conf:
        norm_conf = {'mean': np.zeros(4), 'std': np.ones(4)}
    state_1 = obs_to_state(obs_1, norm_conf, should_normalize=False)
    state_2 = obs_to_state(obs_2, norm_conf, should_normalize=False)
    unnormalized_action = state_2 - state_1
    unnormalized_action[3] = state_2[3] # The gripper action is an absolute action
    return normalize(unnormalized_action, norm_conf['mean'], norm_conf['std'])


def denormalize_action(action, norm_conf):
    """
    :param action: batch of actions with shape (B, a_dim)
    :param norm_conf: dict containing mean and scale parameters
    :return: denormalized actions in real testbench tick space
    """
    return denormalize(action, norm_conf.mean, norm_conf.scale)
