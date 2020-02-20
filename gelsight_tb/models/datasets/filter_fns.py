from gelsight_tb.utils.obs_to_np import denormalize


class InsertFilter:

    def __call__(self, datapoint, conf):
        state = datapoint['state']
        denormalized_state = denormalize(state, conf.norms.state_norm)
        if denormalized_state[0] > 7000 and denormalized_state[2] < 200:
            return True
        return False
