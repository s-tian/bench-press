from bench_press.utils.obs_to_np import denormalize


class InsertFilter:

    def __call__(self, datapoint, conf):
        state = datapoint['state']
        denormalized_state = denormalize(state, conf.norms.state_norm.mean, conf.norms.state_norm.scale)
        if denormalized_state[0] <= 5000 and denormalized_state[2] < 500:
            return True
        return False

class PatternInsertFilter:

    def __call__(self, datapoint, conf):
        state = datapoint['state']
        denormalized_state = denormalize(state, conf.norms.state_norm.mean, conf.norms.state_norm.scale)
        if denormalized_state[0] <= 5000 and denormalized_state[2] < 500:
            return True
        return False
