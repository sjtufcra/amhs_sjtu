from loguru import logger as log


async def output_new(p, v):
    if p.debug_on:
        return None
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
    return None


def output_close_connection(p):
    p.db_cursor.close()
    p.db_connection.close()
    return 0
