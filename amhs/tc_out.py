import pandas as pd
from loguru import logger as log
from time import sleep
import json


def output(p):
    if p.Control().out_path == 0:
        # to local computer
        df = pd.DataFrame(columns=['COMMANDID', 'VEHICLE', 'POSPATH'])
        for k, v in p.orders.items():
            if v.finished == 1:
                df[len(df)] = [k, v.vehicle_assigned, v.delivery_route]
        df.to_excel('./track_use.xlsx', index=False)
    else:
        # to oracle
        n = 0
        for k, v in p.orders.items():
            if v.finished == 1:
                s0 = "SET VEHICLE = '" + v.vehicle_assigned + "', POSPATH = '" + ','.join(v.delivery_route)
                s1 = "' WHERE COMMANDID = '" + k + "'"
                sql = "UPDATE TRANSFER_TABLE " + s0 + s1
                p.db_cursor.execute(sql)
                n += 1
        log.info('Loaded tasks to db number is', n)
        p.db_connection.commit()
        p.db_cursor.close()
        p.db_connection.close()
    return 0


def output_new(p, k, v):
    s0 = "SET VEHICLE = '" + v.vehicle_assigned + "', POSPATH = '" + ','.join(v.delivery_route)
    s1 = "' WHERE COMMANDID = '" + k + "'"
    sql = "UPDATE TRANSFER_TABLE " + s0 + s1
    log.info(f'sql is: {sql}')
    with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute(sql)
            db_conn.commit()
            cursor.close()

    # checking if tasks are assigned successfully
    sleep(0.1)
    idx = "Car:monitor:128.168.11.142_1" + v.vehicle_assigned[1:]
    tmp = json.loads(p.redis_link.get(idx))['ohtStatus_Idle']
    log.info(f'vehicle[{v.vehicle_assigned}],status[{tmp}],1:false/0:true')
    # old
    # p.db_cursor.execute(sql)
    # p.db_connection.commit()
    return 0


def output_close_connection(p):
    p.db_cursor.close()
    p.db_connection.close()
    return 0
