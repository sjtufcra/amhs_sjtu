import redisclusterlocal as rds
import json
pool = rds.ClusterConnectionPool(host='10.34.58.42', port=6379)
connection = rds.RedisCluster(connection_pool=pool)
print(connection)
# 
v = connection.get('Car:location:128.168.11.142_10114')
print(v)
da = json.load(v)
print(da)
