from loguru import logger as log
import time
import json


def output_new(p, v):
    sql0 = (f"UPDATE TRANSFER_TABLE "
            f"SET VEHICLE = \'{v.vehicle_assigned}\', POSPATH = \'{','.join(v.delivery_route)} \'"
            f"WHERE COMMANDID = \'{v.id}\'")
    s0 = "SET VEHICLE = '" + v.vehicle_assigned + "', POSPATH = '" + ','.join(v.delivery_route)
    s1 = "' WHERE COMMANDID = '" + v.id + "'"
    sql = "UPDATE TRANSFER_TABLE " + s0 + s1
    log.info(f'sql is: {sql}')
    with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute(sql)
            db_conn.commit()
            cursor.close()

    # checking if tasks are assigned successfully
    idx = "Car:monitor:128.168.11.142_1" + v.vehicle_assigned[1:]
    tmp = json.loads(p.redis_link.get(idx))['ohtStatus_Idle']
    log.info(f'vehicle[{v.vehicle_assigned}],status[{tmp}],1:false/0:true')
    return None


def output_close_connection(p):
    p.db_cursor.close()
    p.db_connection.close()
    return 0
