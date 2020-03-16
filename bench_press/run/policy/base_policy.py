class BasePolicy:

    def __init__(self, conf):
        self.policy_conf = conf

    def get_action(self, observation, num_steps):
        raise NotImplementedError
