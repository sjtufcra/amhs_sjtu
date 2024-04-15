# server's parameter
httpServer = dict(
    # oracle database linking
    oracle_user='wf2prodmoc',
    oracle_password='wf2moc_22FAB2',
    oracle_dsn='10.34.58.15:1521/f2wdb',
    # redis linking
    rds_connection='10.34.58.42',
    rds_port=6379,
    rds_search_pattern="Car:monitor:*",
    # controlling_on-off
    # if it is equal to 1, this programme will run constantly until being forced to stop
    # or 0, just run once
    circulating_on=1,
    
    # slecet-path 1 or 2
    algorithm_on=1 
)









