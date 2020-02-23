from gelsight_tb.utils.obs_to_np import denormalize


class InsertFilter:

    def __call__(self, datapoint, conf):
        state = datapoint['state']
        denormalized_state = denormalize(state, conf.norms.state_norm.mean, conf.norms.state_norm.scale)
        if denormalized_state[0] <= 4000 and denormalized_state[2] < 400:
            return True
        return False
