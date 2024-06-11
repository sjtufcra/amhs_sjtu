from tc_main import *
import os
import yaml
from loguru import logger as log

runmode = 'runing_mode'
server = 'httpServer'  # online
local = 'localServer'  # local
# config_file_path = './test/config.yaml' #local
config_file_path = './amhs_sjtu/amhs/test/config.yaml' #local


# read config
def read_yaml_config(file_path):
    path = os.path.abspath(file_path)
    with open(path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    return config_data


# check mode
def check_config(config):
    if config.get(runmode) is None:
        return config.get('httpServer')
    else:
        if config.get(runmode) == 1:
            return config.get(server)
        else:
            return config.get(local)


config = read_yaml_config(config_file_path)
mode = check_config(config)
Tc = Amhs(mode)
# logpath
log.add(mode['log'])
Tc.start()
