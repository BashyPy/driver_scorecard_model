from sqlalchemy import create_engine
import pymysql
import pandas as pd
from keys import user, password, host, database, table1, table2, port

host = host
password = password
database = database
port = port
table1 = table1
table2 = table2

db_connection_str = pymysql.connect(
    host=host,
    user=user,
    password=password,
    db=database)

try:
    query = pd.read_sql_query(
        f'''SELECT e.*,
                    p.latitude,
                    p.longitude,
                    p.speed,
                    p.altitude,
                    p.course,
                    p.accuracy,
                    p.network
                    from {table1} as e
                    join {table2} as p
                    on p.id = e.positionid;''',
        db_connection_str
    )

    data = pd.DataFrame(query, columns=['id', 'type', 'eventtime', 'deviceid',
                                        'positionid', 'geofenceid', 'attributes', 'maintenanceid', 'latitude', 'longitude', 'speed', 'altitude', 'course', 'accuracy', 'network'])

    # data = pd.read_sql(
    #    'SELECT e.*, p.latitude, p.longitude, p.speed, p.altitude, p.course, p.accuracy, p.network from tc_events as e join tc_positions as p on p.id = e.positionid;', con=db_connection)

    data['timestamp'] = pd.to_datetime(data['eventtime'])
    data['eventdate'] = data['timestamp'].dt.date
    data['eventtime'] = data['timestamp'].dt.time
    data['eventday'] = data['timestamp'].dt.day_name()
    data['eventyear'] = data['timestamp'].dt.year
    data['eventmonth'] = data['timestamp'].dt.month_name()
    data['eventweek'] = data['timestamp'].dt.isocalendar().week
    data['eventdayofweek'] = data['timestamp'].dt.dayofweek
    data['eventdayofyear'] = data['timestamp'].dt.dayofyear
    data['eventquarter'] = data['timestamp'].dt.quarter
    data['eventhour'] = data['timestamp'].dt.hour

    data_splitted = []
    for i in data['attributes']:
        try:
            data_splitted.append(i.split(":")[1])
        except:
            data_splitted.append('')

    data['event'] = data_splitted
    data['event'] = data['event'].apply(
        lambda x: x.replace('"', ''))
    data['event'] = data['event'].apply(
        lambda x: x.replace("}", ''))
    updated_data = data[['deviceid', 'positionid', 'timestamp', 'event', 'eventdate', 'eventtime', 'eventday', 'eventyear', 'eventmonth', 'eventweek',
                         'eventdayofweek', 'eventdayofyear', 'eventquarter', 'eventhour', 'latitude', 'longitude', 'altitude', 'speed', 'course', 'accuracy', ]]

    updated_data.to_csv('./Blackbox Data/data.csv', index=False, header=True)

    data.to_csv('./Blackbox Data/live_data.csv', index=False, header=True)

    print('Success: data converted successfully')

except:
    print("Error: unable to convert the data")

db_connection_str.close()
