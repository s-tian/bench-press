from bench_press.run.policy.nn_onestep_insert_policy import NNOnestepInsertPolicy


class NNOnestepInsertPolicyOpto(NNOnestepInsertPolicy):
    PRESS_DIST = 1450
    UP_DIST = 250

    def __init__(self, conf):
        super(NNOnestepInsertPolicyOpto, self).__init__(conf)
