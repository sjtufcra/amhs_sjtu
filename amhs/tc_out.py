import pandas as pd


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
        print('Loaded tasks to db number is', n)
        p.db_connection.commit()
        p.db_cursor.close()
        p.db_connection.close()
    return 0


def output_new(p, k, v):
    s0 = "SET VEHICLE = '" + v.vehicle_assigned + "', POSPATH = '" + ','.join(v.delivery_route)
    s1 = "' WHERE COMMANDID = '" + k + "'"
    sql = "UPDATE TRANSFER_TABLE " + s0 + s1
    p.db_cursor.execute(sql)
    p.db_connection.commit()
    return 0


def output_close_connection(p):
    p.db_cursor.close()
    p.db_connection.close()
    return 0
