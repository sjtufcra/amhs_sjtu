import redisclusterlocal as rds

pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
connection = rds.RedisCluster(connection_pool=pool)
v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))