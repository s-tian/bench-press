class BasePolicy:

    def get_action(self, observation, num_steps):
        raise NotImplementedError