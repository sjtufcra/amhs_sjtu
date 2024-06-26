# online
from tc_class_sets import *
from tc_in import *
from tc_optimize import *
from tc_out import *



class Amhs():
    def __init__(self,config) -> None:
        self.runBool = True
        self.config = config
        pass
    
    def start(self):
        d = Dataset(self.config)
        self.Node = d
        d = generating(d)
        d = epoch_static(d)
    
    def over(self):
        output_close_connection(self.d)
        del self.Node
        
    def setRunBool(self,bool=True):
        self.Node.runBool = bool
        if self.Node.runBool == True:
            self.Node.task_assign_new(self.Node)
        else:
            log.info(f'modify run status {self.Node.runBool}')
            pass

    def loadBool(self):
        data = rds.ClusterConnectionPool(host=self.config['rds_connection'], port=self.config['rds_port'])
        connection = rds.RedisCluster(connection_pool=data)
        flag = connection.mget(keys=connection.keys(self.config['rds_search_model']))
        if int(flag) == 4:
            return True
        else:
            return False
