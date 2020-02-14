import numpy as np


def normalize(arr, mean, std):
    return 1.0 * (arr - np.array(mean)) / np.array(std)


def denormalize(arr, mean, std):
    return (arr * np.array(std)) + np.array(mean)


def normalize_img(img):
    #return (img / 255.) * 2 - 1
    return img


def obs_to_state(obs, norm_conf, should_normalize=True):
    x = obs['tb_state']['x']
    y = obs['tb_state']['y']
    z = obs['tb_state']['z']
    dynamixel = obs['dynamixel_state']

    unnormalized_state = np.concatenate((np.array([x, y, z]), np.array([dynamixel])))
    if not should_normalize:
        return unnormalized_state
    normalized_state = normalize(unnormalized_state, norm_conf.mean, norm_conf.scale)
    return normalized_state


def obs_to_images(obs):
    images_dict = obs['images']
    images = []
    for key in sorted(images_dict.keys()):
        images.append(images_dict[key])
    return [normalize_img(image).astype(np.uint8) for image in images]


def obs_to_action(obs_1, obs_2, norm_conf):
    state_1 = obs_to_state(obs_1, norm_conf, should_normalize=False)
    state_2 = obs_to_state(obs_2, norm_conf, should_normalize=False)
    unnormalized_action = state_2 - state_1
    unnormalized_action[3] = state_2[3] # The gripper action is an absolute action
    return normalize(unnormalized_action, norm_conf.mean, norm_conf.scale)


def denormalize_action(action, norm_conf):
    """
    :param action: batch of actions with shape (B, a_dim)
    :param norm_conf: dict containing mean and scale parameters
    :return: denormalized actions in real testbench tick space
    """
    return denormalize(action, norm_conf.mean, norm_conf.scale)
