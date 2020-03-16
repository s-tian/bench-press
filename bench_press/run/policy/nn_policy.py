import bench_press.run.actions.action as action
import numpy as np
import torch
from bench_press.models.datasets.transforms import ImageTransform
from bench_press.models.modules.pretrained_encoder import pretrained_model_normalize
from bench_press.run.policy.base_policy import BasePolicy
from bench_press.run.policy.keyboard_policy import KeyboardPolicy
from bench_press.utils.infra import str_to_class, deep_map
from bench_press.utils.obs_to_np import obs_to_state, obs_to_images, obs_to_opto, denormalize_action
from omegaconf import OmegaConf
from torchvision import transforms


class NNPolicy(BasePolicy):

    def __init__(self, conf):
        super(NNPolicy, self).__init__(conf)
        self.policy_conf.model_conf = OmegaConf.load(self.policy_conf.model_conf_path)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.image_transform = transforms.Compose(
            [
                ImageTransform(transforms.ToPILImage()),
                ImageTransform(transforms.Resize(tuple(self.policy_conf.model_conf.model.final_size))),
                ImageTransform(transforms.ToTensor()),
                ImageTransform(pretrained_model_normalize),
            ])
        self.model_class = str_to_class(self.policy_conf.model_conf.model.type)
        self.model = self.model_class(self.policy_conf.model_conf.model).to(self.device)
        print(f'Loading model from {self.policy_conf.model_checkpoint}')
        checkpoint = torch.load(self.policy_conf.model_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.keyboard_override = False
        self.keyboard_policy = KeyboardPolicy(None)

    @staticmethod
    def prep_images(images):
        """
        :param images:
        :return: prepped images, with batch dimension added and channel dimension switched
        """
        prepped = []
        for im in images:
            im_p = np.transpose(im, (2, 0, 1)).astype(np.float32)
            prepped.append(im_p)
        return prepped

    def forward_model(self, observation, press_obs=None):
        action_norm = self.policy_conf.model_conf.dataset.norms.label
        state_norm = self.policy_conf.model_conf.dataset.norms.state

        print(f'state coming in is {observation["tb_state"]}')
        state = obs_to_state(observation, state_norm).astype(np.float32)
        inp = {
            'state': state[None],  # Add observation batch dimension to state
        }

        if self.policy_conf.optoforce:
            if self.policy_conf.optoforce:
                opto_press_norm = self.policy_conf.model_conf.dataset.norms.opto_1
                opto_curr_norm = self.policy_conf.model_conf.dataset.norms.opto_2
            opto_1 = obs_to_opto(press_obs, opto_press_norm).astype(np.float32)
            opto_2 = obs_to_opto(observation, opto_curr_norm).astype(np.float32)
            inp['opto_1'], inp['opto_2'] = opto_1[None], opto_2[None]  # Add batch dimension here
        elif press_obs:
            observation['raw_images']['gelsight_top'] = press_obs['raw_images']['gelsight_top']
            observation['images']['gelsight_top'] = press_obs['images']['gelsight_top']

        images = obs_to_images(observation)
        inp['images'] = images

        inp = self.image_transform(inp)
        inp = deep_map(lambda x: torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x, inp)
        inp = deep_map(lambda x: x.to(self.device), inp)
        for i, img in enumerate(inp['images']):
            inp['images'][i] = img[None]
        output = self.model(inp).cpu().detach().numpy()
        print(f'normalized output: {output}')
        output = denormalize_action(output, action_norm)[0]
        print(f'denormalized output: {output}')

        gripper_open = True
        if output[-1] < -49.5 / 2:
            gripper_open = False
        gripper_action = action.DynamixelAngleAction(0) if gripper_open else action.DynamixelAngleAction(-49.5)
        return output[:3], gripper_action

    def get_action(self, observation, num_steps, press_obs=None):
        if num_steps == 0:
            self.keyboard_override = False
        if not self.keyboard_override:
            response = input('take control?')
            if response == 'y':
                self.keyboard_override = 1000
        if self.keyboard_override:
            self.keyboard_override -= 1
            return self.keyboard_policy.get_action(observation, num_steps)

        xyz_action, gripper_action = self.forward_model(observation, press_obs)

        if self.policy_conf.order:
            build_action = []
            assert isinstance(self.policy_conf.order, str) and len(
                self.policy_conf.order) <= 3, "Order param must be in format of permutation of x, y, z"
            for axis in self.policy_conf.order:
                if axis == 'x':
                    build_action.append(action.DeltaAction((xyz_action[0], 0, 0)))
                elif axis == 'y':
                    build_action.append(action.DeltaAction((0, xyz_action[1], 0)))
                elif axis == 'z':
                    build_action.append(action.DeltaAction((0, 0, xyz_action[2])))
            return action.SequentialAction(build_action)

        return action.SequentialAction(
            [
                action.DeltaAction(xyz_action),
                gripper_action,
            ]
        )
