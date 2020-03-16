from bench_press.run.policy.base_policy import BasePolicy


class SequentialPolicy(BasePolicy):
    # This class is for the creation of 'sequential' policies, which essentially just means that each action
    # in sequence known ahead of time and not really dependent on observations, for example, when executing a
    # fixed routine. In this situation it can be simpler to supply the actions in terms of an iterator
    # (in this case, implemented as a lazy generator function), which can be done by subclassing this class and
    # overriding `action_generator`.

    def __init__(self, config):
        super(SequentialPolicy, self).__init__(config)
        self.action_iterator = None

    def get_action(self, observation, num_steps):
        if num_steps == 0:  # At the beginning of every trajectory, reset the policy
            self.action_iterator = self.action_generator()
        action = self.action_iterator.next()
        return action

    def action_generator(self):
        raise NotImplementedError
