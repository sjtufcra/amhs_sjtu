from .tc_class_sets import *
from .tc_in import *
from .tc_assign import *
from .tc_out import *

class Amhs():
    def __init__(self,config) -> None:
        self.runBool = True
        self.config = config
        pass
    
    # 开启数据服务
    def start(self):
        d = Dataset(self.config)
        self.Node = d
        if d.pattern == 1:
            d = generating(d)
            d = task_assign_new(d)
        elif d.pattern == 0:
            d = generating(d)
            d = task_assign_new(d)
    
    # 关闭数据库
    def over(self):
        output_close_connection(self.d)
        
    # 起停数据库读取
    def setRunBool(self,bool=True):
        self.Node.runBool = bool

