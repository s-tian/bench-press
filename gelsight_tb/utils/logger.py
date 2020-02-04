from pathlib import Path
from omegaconf import OmegaConf
import datetime
import deepdish as dd


class Logger:

    def __init__(self, conf):
        self.all_conf = conf
        self.logger_conf = conf.logger
        self.log_dir = self._make_log_dir()
        self._log_conf()

    def _log_conf(self):
        OmegaConf.save(self.all_conf, f"{self.log_dir}/conf.yaml")

    def _make_log_dir(self):
        logs_folder = self.logger_conf.log_folder
        current_time = datetime.datetime.now().strftime('%m-%d:%H:%M:%S')
        log_dir_path = f"{logs_folder}/{current_time}"
        Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        return log_dir_path

    def log_text(self, text):
        if self.logger_conf.log_text:
            print(text)

    def log_obs(self, obs, index):
        assert isinstance(obs, list), 'observations should be logged in list format, first dim being time'
        dd.io.save(f"{self.log_dir}/record_{index}.h5", obs)


