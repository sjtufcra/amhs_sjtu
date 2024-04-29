import concurrent.futures
import itertools

def calculate_sum(pair,no):
    """计算一对元素的和"""
    x, y = pair
    return x + y

def process_data(data):
    """
    使用多线程遍历数据列表，计算每对元素的和。
    
    :param data: 长度为2000的列表
    """
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用itertools.combinations生成所有可能的元素对
        element_pairs = itertools.combinations(data, 2)
        
        # 将计算任务提交给线程池
        results = list(executor.map(calculate_sum, element_pairs))
        
    # 打印结果或根据需要处理结果
    for result in results:
        print(result)

# 示例数据
data_example = [i for i in range(10)]

# 调用函数
process_data(data_example)