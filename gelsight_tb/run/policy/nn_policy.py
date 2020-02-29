import torch
from torchvision import transforms
import numpy as np
from omegaconf import OmegaConf

from gelsight_tb.run.policy.base_policy import BasePolicy
from gelsight_tb.run.policy.keyboard_policy import KeyboardPolicy 
from gelsight_tb.utils.obs_to_np import obs_to_state, obs_to_images, denormalize_action
from gelsight_tb.utils.infra import str_to_class, deep_map
from gelsight_tb.models.datasets.transforms import ImageTransform
from gelsight_tb.models.modules.pretrained_encoder import pretrained_model_normalize
import gelsight_tb.run.actions.action as action


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

    def forward_model(self, observation):
        action_norm = self.policy_conf.model_conf.dataset.norms.action_norm
        state_norm = self.policy_conf.model_conf.dataset.norms.state_norm
        images = obs_to_images(observation)
        images = self.prep_images(images)

        state = obs_to_state(observation, state_norm).astype(np.float32)
        inp = {
            'images': images,
            'state': state[None] # Add batch dimension to state
        }
        inp = deep_map(lambda x: torch.from_numpy(x), inp)
        inp = self.image_transform(inp)
        inp = deep_map(lambda x: x.to(self.device), inp)
        for i, img in enumerate(inp['images']):
            inp['images'][i] = img[None]
        #output = denormalize_action(self.model(inp).cpu().detach().numpy(), action_norm)[0]
        output = self.model(inp).cpu().detach().numpy()
        print(f'normalized output: {output}')
        output = denormalize_action(output, action_norm)[0]
        print(f'denormalized output: {output}')

        gripper_open = True
        if output[-1] < -49.5 / 2:
            gripper_open = False
        gripper_action = action.DynamixelAngleAction(0) if gripper_open else action.DynamixelAngleAction(-49.5)
        for i in range(len(output)):
            if np.abs(output[i]) < 5:
                output[i] = 0
        return output[:3], gripper_action

    def get_action(self, observation, num_steps):
        if num_steps == 0:
            self.keyboard_override = False
        if not self.keyboard_override:
            response = input('take control?')
            if response == 'y':
                self.keyboard_override = 1000
        if self.keyboard_override:
            self.keyboard_override -= 1
            return self.keyboard_policy.get_action(observation, num_steps)

        xyz_action, gripper_action = self.forward_model(observation)

        if self.policy_conf.order:
            build_action = []
            assert isinstance(self.policy_conf.order, str) and len(self.policy_conf.order) <= 3, "Order param must be in format of permutation of x, y, z"
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


