import json
import itertools

class Configuration():
    def __init__(self):
        self.congif_path='configuration\config.json'
        self.parameter_dict_path='configuration\model_parameters.json'

    def get_config(self):
        with open(self.congif_path, "r") as f:
            self.config_json = json.load(f)

            return self.config_json


    def get_experiments_list(self):
        with open(self.parameter_dict_path, "r") as f:
            self.parameter_dict = json.load(f)

        keys, values = zip(*self.parameter_dict.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return experiments
