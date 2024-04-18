from .tc_class_sets import *
from .tc_in import *
from .tc_assign import *
from .tc_out import *
from loguru import logger as log
class Amhs():
    def __init__(self,config) -> None:
        self.runBool = True
        self.config = config
        pass
    
    def start(self):
        d = Dataset(self.config)
        self.Node = d
        log.info(f'Enable algorithmic scheduling')
        d = generating(d)
        d = task_assign_new(d)
    
    def over(self):
        log.info(f'Close the database connection and clear the cache')
        output_close_connection(self.d)
        del self.Node
        
    def setRunBool(self,bool=True):
        self.Node.runBool = bool
        if self.Node.runBool == True:
            self.Node.task_assign_new(self.Node)
            log.info(f'modify run status {self.Node.runBool}')
        else:
            log.info(f'modify run status {self.Node.runBool}')
            pass

