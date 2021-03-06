from bench_press.run.actions.action import SleepAction
from bench_press.run.policy.base_policy import BasePolicy


class DummyPolicy(BasePolicy):

    def get_action(self, observation, num_steps):
        # print(f'Policy got obs {observation}')
        # print(f'Step: {num_steps}')
        return SleepAction(1)
