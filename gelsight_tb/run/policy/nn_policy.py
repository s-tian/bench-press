import torch
from torchvision import transforms
import numpy as np

from gelsight_tb.run.policy.base_policy import BasePolicy
from gelsight_tb.utils.obs_to_np import obs_to_state, obs_to_images, denormalize_action
from gelsight_tb.utils.infra import str_to_class, deep_map
from gelsight_tb.models.datasets.transforms import ImageTransform
from gelsight_tb.models.modules.vgg_encoder import pretrained_model_normalize
import gelsight_tb.run.actions.action as action


class NNPolicy(BasePolicy):

    def __init__(self, conf):
        super(NNPolicy, self).__init__(conf)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.image_transform = transforms.Compose(
            [
                ImageTransform(transforms.ToPILImage()),
                ImageTransform(transforms.Resize(64)),
                ImageTransform(transforms.ToTensor()),
                ImageTransform(pretrained_model_normalize),
            ])
        self.model_class = str_to_class(self.policy_conf.model_conf.model.type)
        self.model = self.model_class(self.policy_conf.model_conf.model).to(self.device)
        print(f'Loading model from {self.policy_conf.model_checkpoint}')
        checkpoint = torch.load(self.policy_conf.model_checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

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

    def get_action(self, observation, num_steps):
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
        import ipdb; ipdb.set_trace()
        output = denormalize_action(output, action_norm)[0]

        gripper_open = True
        if output[-1] < -49.5 / 2:
            gripper_open = False
        gripper_action = action.DynamixelAngleAction(0) if gripper_open else action.DynamixelAngleAction(-49.5)
        for i in range(len(output)):
            if np.abs(output[i]) < 10:
                output[i] = 0

        return action.SequentialAction(
            [
                action.DeltaAction(output[:3]),
                gripper_action,
            ]
        )


