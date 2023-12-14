import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Union
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as st
import seaborn as sns
import starlette.responses as responses
from dotenv import load_dotenv
import pymysql
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from scipy.stats import rankdata
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic_models import Token, TokenData, User, UserInDB


from users import users_db
from error_logger import logger

# Suppress scientific notation
pd.options.display.float_format = '{:.2f}'.format
pd.options.plotting.backend = "plotly"
sns.set(style="ticks", font_scale=1.1)
warnings.filterwarnings("ignore")


paths = ['csv files', 'HTML Charts', 'HTML Charts/Year',
         'HTML Charts/Year and Month', 'HTML Charts/Year and Month and ID',
         'HTML Charts/Year and Quarter', 'HTML Charts/Year and Quarter and ID', ]


for path in paths:
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    else:
        pass


load_dotenv()

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

host = os.getenv('host')
user = os.getenv('user')
database = os.getenv('database')
password = os.getenv('password')
table1 = os.getenv('table1')
table2 = os.getenv('table2')
table3 = os.getenv('table3')
table4 = os.getenv('table4')

# Specify the schema (replace 'your_schema' with the actual schema name)
schema_name = f'{database}'

# Create a SQLAlchemy engine
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
# Create a MetaData object with the specified schema
metadata = MetaData()

metadata.bind = engine

# Pass the metadata to the read_sql_query function
try:
    global query_
    query = pd.read_sql_query(
        f'''SELECT e.*,
        p.latitude,
        p.longitude,
        p.speed,
        p.altitude,
        p.course,
        p.accuracy,
        ud.userid AS 'Company ID',
        u.name AS 'CompanyName',
        u.email AS Email
        FROM {table1} AS e
        JOIN {table2} AS p
        ON p.id = e.positionid
        JOIN {table3} AS ud
        ON ud.deviceid = e.deviceid
        JOIN {table4} AS u
        ON u.id = ud.userid
        WHERE e.type = 'alarm';''',
        engine,
        index_col=None,  # Specify your index_col if needed
        params=None,     # Specify any additional parameters if needed
        coerce_float=True,  # Specify whether to coerce float columns
        parse_dates=None,   # Specify date columns to parse
        chunksize=None      # Specify chunksize for reading in chunks
    )
    query_ = query

    data = pd.DataFrame(query_,
                        columns=['id', 'type', 'eventtime', 'deviceid',
                                 'positionid', 'geofenceid', 'attributes',
                                 'maintenanceid', 'latitude', 'longitude',
                                 'altitude', 'speed', 'course', 'accuracy',
                                 'network', 'Company ID', 'Company Name',
                                 'Email'])

    data['timestamp'] = pd.to_datetime(data['eventtime'])
    data['eventdate'] = data['timestamp'].dt.date
    data['eventtime'] = data['timestamp'].dt.time
    data['eventday'] = data['timestamp'].dt.day_name()
    data['eventyear'] = data['timestamp'].dt.year
    data['eventmonth'] = data['timestamp'].dt.month_name()
    data['eventweek'] = data['timestamp'].dt.isocalendar().week
    data['eventdayofweek'] = data['timestamp'].dt.dayofweek
    data['eventdayofmonth'] = data['timestamp'].dt.day
    data['eventdayofyear'] = data['timestamp'].dt.dayofyear
    data['eventquarter'] = data['timestamp'].dt.quarter
    data['eventhour'] = data['timestamp'].dt.hour
    data_splitted = []

    for i in data['attributes']:
        try:
            data_splitted.append(i.split(":")[1])
        except Exception as e:
            logger.warning(f"Error: {e}")
            logger.error('Exception occurred', exc_info=True)
            data_splitted.append('')

    data['event'] = data_splitted
    data['event'] = data['event'].apply(lambda x: x.replace('"', ''))
    data['event'] = data['event'].apply(lambda x: x.replace("}", ''))
    data['DriverID'] = data['deviceid']

    def part_of_day(x):
        if (x > 4) and (x <= 8):
            return 'Early Morning'
        elif (x > 8) and (x <= 12):
            return 'Morning'
        elif (x > 12) and (x <= 16):
            return 'Noon'
        elif (x > 16) and (x <= 20):
            return 'Eve'
        elif (x > 20) and (x <= 24):
            return 'Night'
        elif (x <= 4):
            return 'Late Night'

    data['eventpartofday'] = data['eventhour'].apply(
        part_of_day)

    def test():
        global updated_data_

        updated_data = data[
            ['DriverID', 'Company ID', 'Company Name', 'Email', 'positionid',
             'timestamp', 'event', 'eventdate', 'eventtime', 'eventday',
             'eventyear', 'eventmonth', 'eventweek', 'eventdayofweek',
             'eventdayofmonth', 'eventdayofyear', 'eventquarter', 'eventhour',
             'eventpartofday', 'latitude', 'longitude', 'altitude', 'speed',
             'course', 'accuracy', ]]

        updated_data_ = updated_data

    test()

    updated_data_.to_csv(
        './csv files/data.csv', index=False, header=True,
        encoding='utf-8')
    data.to_csv(
        './csv files/main_data.csv', index=False, header=True,
        encoding='utf-8')

    # result = updated_data_.to_json(orient="records")
    # parsed = json.loads(result)
    # json.dumps(parsed, indent=4)
except Exception as e:
    print(f"Error: {e}")
    logger.warning(f"Error: {e}")
    logger.error('Exception occurred', exc_info=True)


# Create declarative base meta instance
Base = declarative_base()
# Create session local class for session maker
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(user_db, username: str, password: str):
    user = get_user(user_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(
        data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        # credentials_exception
        logger.warning("Could not validate credentials")
        logger.error('Exception occurred', exc_info=True)
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
        current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.get('/', response_model=User)
async def index(current_user: User = Depends(get_current_active_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return responses.RedirectResponse("/docs")


@app.post("/token", response_model=Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.get('/fetchData')
async def get_all_data():

    result = updated_data_.to_json(orient="records")
    parsed = json.loads(result)
    json.dumps(parsed, indent=4)
    return {"message": "Success", "status": 200}


@app.get('/fetchData/{DriverID}')
async def get_data_per_driver(DriverID: int):
    if DriverID in updated_data_['DriverID'].values:
        data = updated_data_[updated_data_['DriverID'] == DriverID]
        result = data.to_json(orient="records")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        return {"message": "Success", "status": 200, "data": result}
    else:
        return {"message": "Driver not found", "status": 404}


@app.get("/getMonthlyData/{month}")
async def get_data_per_month(month: str):
    if month in updated_data_['eventmonth'].values:
        data = updated_data_[updated_data_['eventmonth'] == month]

        if data.shape[0] < 1:
            return {"message": "Not Enough Records", "status": 401}
        result = data.to_json(orient="records")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        return {"message": "Success", "status": 200, "data": result}
    return {"message":
            "Not enough record for this month. Please select another month",
            "status": 401}


@app.get("/getMonthlyDataPerDriver/{month}/{DriverID}")
async def get_data_per_month_per_driver(month: str, DriverID: int):
    if month in updated_data_['eventmonth'].values:
        data = updated_data_[(updated_data_['eventmonth'] == month) & (
            updated_data_['DriverID'] == DriverID)]

        if data.shape[0] < 1:
            return {"message": "Not Enough Records", "status": 401}
        result = data.to_json(orient="records")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        return {"message": "Success", "status": 200, "data": result}
    return {"message":
            "Not enough record for this month. Please select another month",
            "status": 401}


@app.get('/getAnnualScore/{year}')
async def get_driver_score_per_year(year: int):
    data = updated_data_
    day = data[data['eventyear'] == year]

    if day.shape[0] < 600:
        return {"message":
                "Not Enough Records, Please Select Another Month",
                "status": 401}
    else:
        day = data[data['eventyear'] == year]
        # print('A preview at the selected month', day.head())
        # eventCount = len(day.event)
        # print(
        # f'
        # The number of events made in the selected month is: {eventCount}
        # events')
        # driverCount = day.DriverID.nunique()
        # print(
        # f"The number of drivers in this date range is: {driverCount}")
        eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            {"event": "count"}).add_prefix('Number of ')
        # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
        # print(
        # f'The average number of events made by the drivers is: {averageNoEvents}')

        fig = px.bar(eventsPerDriver.reset_index(),  x='Number of event',
                     y='DriverID', color='Number of event', barmode='group', orientation='h')
        fig.update_layout(
            yaxis_title="Number of Events",
            xaxis_title="driver ID",
            legend_title="DriverID",
            title="Bar Chart of Trips made per driver",
            template="plotly_white",
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Number of trips per Driver.html')

        fig = px.histogram(eventsPerDriver.reset_index(),
                           x='Number of event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title="Histogram of Event Counts",
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Histogram of Events.html')

        # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
        # DriverID = eventsPerDriver['Number of event'].idxmax()
        # print(
        # f'The driver with the most events is: driver {DriverID}, and the number of events made is: {maxEventsPerDriver}')
        # minEventsPerdriver = eventsPerDriver['Number of event'].min()
        # DriverID = eventsPerDriver['Number of event'].idxmin()
        # print(
        # f'The driver with the least events is: driver {DriverID}, and the number of events made is: {minEventsPerdriver}')
        # Event Type
        dfReasonHist = day.groupby(['event'])[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=True)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Count of Events",
            title='Bar Plot of Event Distribution',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Bar Plot of Event Distribution.html')

        # Handling Behavioral and Non-behavioral Events
        # non-behavioral events
        # non_behavioral_events = [event for event in day.event if event not in [
        #    'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # newCount = len(day[day['event'].isin(non_behavioral_events)])
        # print("Number of events before removing non-behavioral events is: {}.\nAfter removing non-behavioral events, we have: {} events.\nThis led to a reduction in the data size by: {:0.2f}%, leaving: {:0.2f}% of the entire data size.\nCurrent number of events is: {}".format(
        # len(day), newCount, ((len(day) - newCount)/len(day))*100, (100-(((len(day) - newCount)/len(day))*100)), newCount))
        # Specifying behavioral events
        behavioral_events = [event for event in day.event if event in [
            'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # Initializing the number of minimum events to 50 (intuitively)

        def prepData(day, minRecordsPerSubscriber=0):
            day.reset_index(inplace=True)
            # print(
            # f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
            # Remove NAs
            # df = day.dropna()
            # print(f"Removing NAs, we are left with: {len(df)} trips")
            # Filter out unwanted events
            df4 = day[day['event'].isin(behavioral_events)]
            # print(
            # f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
            # Filter out users with too few samples
            eventCountPerdriver = df4.groupby(
                'DriverID')['DriverID'].agg('count')
            driversWithManyRecords = eventCountPerdriver[eventCountPerdriver >
                                                         minRecordsPerSubscriber]
            driversWithManyRecords.keys()
            df5 = df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            # print(
            # f"Filtering users with too few samples,  we are left with: {len(df5)} trips")
            # print("*** Done. ***")
            return (df5)

        df6 = prepData(day)
        relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
            {"event": "count"}).sort_values(by='event', ascending=False)

        fig = px.histogram(relevantEventsPerSubscriber, x='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title='Histogram of Event Counts',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Histogram of Event Counts.html')

        # Distribution of Events
        dfReasonHist = df6.groupby('event')[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=False)
        # print('Distribution of Events (in descending order)')
        # print(dfReasonHist)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Event",
            title='Distribution of Behavioral Events',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Behavioral Events.html')

        # Hard Acceleration Visual
        hard_acceleration = day[day['event'] == 'hardAcceleration']
        ha = hard_acceleration[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ha = ha.reset_index()
        ha = ha.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Acceleration Events per Driver')
        # print(ha)
        # print("")

        fig = px.bar(ha, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Acceleration per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Hard Acceleration per Driver.html')

        # Hard Braking Visual
        hard_braking = day[day['event'] == 'hardBraking']
        hb = hard_braking[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hb = hb.reset_index()
        hb = hb.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Braking Events per Driver')
        # print(hb)
        # print("")

        fig = px.bar(hb, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Braking per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Hard Braking per Driver.html')

        # Hard Cornering Visual
        hard_cornering = day[day['event'] == 'hardCornering']
        hc = hard_cornering[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hc = hc.reset_index()
        hc = hc.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Cornering Events per Driver')
        # print(hc)
        # print("")

        fig = px.bar(hc, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Cornering per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Hard Cornering per Driver.html')

        # Overspeed Visual

        overspeed = day[day['event'] == 'overspeed']
        ovs = overspeed[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ovs = ovs.reset_index()
        ovs = ovs.sort_values(by='Number of event', ascending=False)
        # print('The number of Overspeeding Events per Driver')
        # print(ovs)
        # print("")

        fig = px.bar(ovs, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Overspeeding per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Overspeeding per Driver.html')

        # Calculate distance traveled in each trip using Haversine

        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        def total_distance(oneDriver):
            dist = haversine(oneDriver.longitude.shift(1), oneDriver.latitude.shift(1),
                             oneDriver.loc[1:, 'longitude'], oneDriver.loc[1:, 'latitude'])
            return np.sum(dist)

        # Calculate the overall distance made per driver

        def calculate_overall_distance_traveled(dfRaw):
            dfDistancePerdriver = day.groupby('DriverID').apply(
                total_distance).reset_index(name='Distance')
            return dfDistancePerdriver
        distancePerdriver = calculate_overall_distance_traveled(df6)
        # print('Distance Traveled per Driver (in descending order)')
        # print(distancePerdriver.sort_values(by='Distance', ascending=False))

        fig = px.bar(distancePerdriver.sort_values(by='Distance', ascending=True),
                     x='DriverID', y='Distance', color='Distance', )
        fig.update_layout(
            xaxis_title="Driver ID",
            yaxis_title="Distance Traveled",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            title='Distance Traveled per Driver',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Distance Traveled per Driver.html')

        # Feature Engineering
        # Transform the events data frame to a features data frame (column for each type of relevant event)

        def create_feature_set(df6, distancePerdriver):
            dfEventAggByDriver = df6.groupby(['DriverID', 'event'])[['event']].agg(
                'count').add_prefix('Number of ').reset_index()
            # Pivot events into columns
            # Pivot the table by setting the drivers' name as the index column, while the respective events takes on a column each.
            # Finally, fill missing observations with zeros(0s)
            dfEventMatrix = dfEventAggByDriver.pivot(index='DriverID', columns=[
                                                     'event'], values='Number of event').add_prefix('F_').fillna(0).reset_index()
            # Merge the created pivot table with the earlier created dataframe for distance traveled per driver.
            dfEventMatrix = dfEventMatrix.merge(
                distancePerdriver, how='inner', on='DriverID')
            dfEventMatrix.set_index('DriverID', inplace=True)
            # Set the columns to start with F_
            featureCols = [
                col for col in dfEventMatrix if col.startswith('F_')]
            # Divide each of the features by the traveled.
            dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                dfEventMatrix['Distance'], axis=0)
            dfFeatureSet = dfEventMatrix[featureCols]
            return dfFeatureSet

        features = create_feature_set(df6, distancePerdriver)

        # Motion based events
        try:
            if 'F_hardAcceleration' in features:
                features['F_hardAcceleration'] = features['F_hardAcceleration']
            else:
                features['F_hardAcceleration'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardBraking' in features:
                features['F_hardBraking'] = features['F_hardBraking']
            else:
                features['F_hardBraking'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardCornering' in features:
                features['F_hardCornering'] = features['F_hardCornering']
            else:
                features['F_hardCornering'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_overspeed' in features:
                features['F_overspeed'] = features['F_overspeed']
            else:
                features['F_overspeed'] = pd.Series(
                    [0 for x in range(len(features.index))])
            # features = [['F_hardAcceleration', 'F_hardBraking', 'F_hardCornering', 'F_overspeed']]
            features = features.fillna(0)
            # print(features)
        except Exception as e:
            print('Error')
            logger.warning(e)
            logger.error('Exception occurred', exc_info=True)

        features = features.rename(columns={'F_hardAcceleration': "Hard Acceleration",
                                            'F_hardBraking': "Hard Braking",
                                            'F_hardCornering': "Hard Cornering",
                                            'F_overspeed': 'Overspeed'}, inplace=False)
        # print(features.head())

        # Driver with the lowest harsh braking score
        # print(
        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmin()])

        # Driver with the highest harsh braking score
        # print(

        #    f'The information about the driver with the least hard braking score is given below: ',

        #    features.loc[features['Hard Acceleration'].idxmax()])

        fig = px.scatter_matrix(features.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()  # type: ignore
        fig.write_html('./HTML Charts/Year/scatter_matrix_features.html')

        # Detecting and Handling Outliers
        # Log Transformation
        features_log = np.log1p(features)
        features_log = features.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        features_log = features.fillna(0)
        # print('A preview of the data upon the application of Log Transformation')
        # print(features_log.head())

        fig = px.scatter_matrix(features_log.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/scatter_matrix_log_transformation.html')

        # Box-Cox Transformation
        # print('Applying Box-Cox transformation on the data')

        def transform_to_normal(x, min_max_transform=False):
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            valueGreaterThanZero = np.where(x <= 0, 0, x)
            positives = x[valueGreaterThanZero == 1]
            if (len(positives) > 0):
                xt[valueGreaterThanZero == 1], _ = st.boxcox(positives+1)
            if min_max_transform:
                xt = (xt - np.min(xt)) / (np.max(xt)-np.min(xt))
            return xt
        # features_1 = features_1.set_index('DriverId')
        transFeatures = features.apply(lambda x: (
            transform_to_normal(x, min_max_transform=True)))
        transFeatures = transFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeatures = transFeatures.fillna(0)
        # print('A preview of the data upon the application of Box-Cox Transformation')
        # print(transFeatures.head())

        fig = px.scatter_matrix(transFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking',
                                                                         'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html('./HTML Charts/Year/scatter_matrix_box_cox.html')

        # Standard Deviation Rule
        # print('Checking for and replacement of outliers')

        def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
            print(x.name)
            x = x.values
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            xt = transform_to_normal(x)
            xMean, xStd = np.mean(xt), np.std(xt)
            outliers = np.where(xt > xMean + stdFactor*xStd)[0]
            inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
            if len(outliers) > 0:
                # print("found outlier with factor: " +
                # str(stdFactor)+" : "+str(outliers))
                xinline = x[inliers]
                maxInRange = np.max(xinline)
                # print("replacing outliers {} with max={}".format(
                # outliers, maxInRange))
                vals = x.copy()
                vals[outliers] = maxInRange
                x = pd.Series(vals)
            else:
                pass
            if normalize:
                # Normalize to [0,1]
                x = (x - np.min(x)) / (np.max(x)-np.min(x))
            return x
        # features_1 = features_1.reset_index()
        cleanFeatures = features.apply(
            lambda x: (replace_outliers_with_limit(x)))
        cleanFeatures = cleanFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        cleanFeatures = cleanFeatures.fillna(0)
        # print("A preview of the cleaned data after handling outliers")
        # print(cleanFeatures.head())

        fig = px.scatter_matrix(cleanFeatures.reset_index(), dimensions=['Hard Acceleration',
                                                                         'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/scatter_matrix_cleaned_features.html')

        # Correlation between events
        # print("Correlation between the events")
        corr = cleanFeatures.corr()
        corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
        corr = corr.fillna(0)
        # print(corr)

        # Plot the heatmap of the correlation matrix
        # print(f"""
        #               ### Correlation Heatmap
        #                """)
        fig = px.imshow(corr, color_continuous_scale='hot',
                        title='Correlation Heatmap', width=600, height=500, aspect='equal')
        # fig.show()
        fig.write_html('./HTML Charts/Year/correlation heatmap.html')

        # Pre step: Normalize features
        # print("Data normalization")
        minPerFeature = cleanFeatures.min()
        maxPerFeature = cleanFeatures.max()
        # print("Min and Max values per column before normalization")
        # for col in range(0, len(cleanFeatures.columns)):
        # print(
        # f"{cleanFeatures.columns[col]} range:[{minPerFeature[col]},{maxPerFeature[col]}]")
        normalizedFeatures = (cleanFeatures-minPerFeature) / \
            (maxPerFeature-minPerFeature)
        normalizedFeatures = normalizedFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        normalizedFeatures = normalizedFeatures.fillna(0)
        # print("A preview of the normalized data")
        # print(normalizedFeatures.head())

        # Standardize features after box-cox as well.
        # print("Standardizing the features after Box-Cox Transformation")
        transFeaturesScaled = (
            transFeatures - transFeatures.mean())/transFeatures.std()
        transFeaturesScaled = transFeaturesScaled.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeaturesScaled = transFeaturesScaled.fillna(0)
        # print("A preview of the standardized data")
        # print(transFeaturesScaled.head())

        # print("Mean and STD before standardization")
        # for col in range(0, len(transFeatures.columns)):
        # print(
        # f"{transFeatures.columns[col]} range:[{transFeatures.mean()[col]},{transFeatures.std()[col]}]")

        # Anomaly Detection:
        # LOF - Local Outlier Filter
        # X = transFeaturesScaled.values
        # X = np.nan_to_num(X)
        # clf = LocalOutlierFactor(n_neighbors=5)
        # isOutlier = clf.fit_predict(X)
        # plt.title("Local Outlier Factor (LOF)", fontsize=20)
        # a = plt.scatter(X[isOutlier == 1, 0], X[isOutlier == 1, 1], c='white',
        #                edgecolor='k', s=40)
        # b = plt.scatter(X[isOutlier == -1, 0], X[isOutlier == -1, 1], c='red',
        #                edgecolor='k', s=40)
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # plt.xlabel(normalizedFeatures.columns[0], fontsize=15)
        # plt.ylabel(normalizedFeatures.columns[1], fontsize=15)
        # plt.xlim((-0.01, 1.01))
        # plt.ylim((-0.01, 1.01))
        # plt.legend([a, b],
        #           ["normal observations",
        #            "abnormal observations"],
        #           loc="upper right", prop={'size': 15}, frameon=True)
        # fig.show()
        # fig.write_html('./HTML Charts/Year/lof.html')

        # Multivariate analysis
        # Dimensionality reduction
        # PCA
        # pca = PCA(n_components=4)
        # principalComponents = pca.fit_transform(normalizedFeatures)
        # column_names = ['principal component {}'.format(
        #    i) for i in range(normalizedFeatures.shape[1])]
        # plt.bar(x=column_names, height=pca.explained_variance_ratio_)
        # plt.title("Percentage of explained variance")
        # fig.show()
        # print("Principal components explained variance ratio: {}.".format(
        # pca.explained_variance_ratio_))
        # principalDf = pd.DataFrame(
        #    data=principalComponents, columns=column_names)
        # df = normalizedFeatures
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # Show correlation matrix of the original features and the first principal component
        # pcAndOriginal = pd.concat(
        #    [principalDf.iloc[:, 0].reset_index(drop=True), normalizedFeatures], axis=1)
        # sns.set(style="ticks")
        # histplot = pcAndOriginal['principal component 0'].hist(figsize=(5, 5))
        # histplot.set_title("principal component 0 histogram")
        # sns.pairplot(pcAndOriginal, y_vars=['principal component 0'],
        #             x_vars=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'])
        # fig.write_html('./HTML Charts/Year/correlation_matrix_pca.html')

        # Extract statistics from the fitted distributions
        # print(normalizedFeatures.head())

        # Fit exponential distribution
        def fit_distribution_params(series):
            # print("Extracting distribution parameters for feature: " +
            #      series.name + " (" + str(len(series)) + ' values)')
            xPositive = series[series > 0]
            xPositive = xPositive.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            xPositive = xPositive.fillna(0)
            # xPositive = xPositive.replace
            probs = np.zeros(len(series))
            if (len(xPositive) > 0):
                params = st.expon.fit(xPositive)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # print('params = {}, {}, {}.'.format(arg, loc, scale))
                return arg, loc, scale

        def calculate_score_for_series(x, fittedParams, verbose=False):
            # print("Calculating scores for feature: " + x.name)
            xPositive = x[x > 0]
            probs = np.zeros(len(x))
            if (len(xPositive) > 0):
                arg = fittedParams[x.name]['arg']
                loc = fittedParams[x.name]['loc']
                scale = fittedParams[x.name]['scale']
                probs[x > 0] = st.expon.cdf(
                    xPositive, loc=loc, scale=scale, *arg)
                if verbose:
                    probs_df = pd.DataFrame(
                        {'Event value': x.values.tolist(), 'Event probability': probs}, index=True)
                    probs_df = probs_df.sort_values(by='Event value')
                    # print(probs_df)
            return probs

        # Store each fitted distribution parameters for later use
        fittedParams = {}
        for col in features.columns:
            arg, loc, scale = fit_distribution_params(features[col])
            fittedParams[col] = {}
            fittedParams[col]['arg'] = arg
            fittedParams[col]['loc'] = loc
            fittedParams[col]['scale'] = scale
        # print('Fitted parameters:')
        # print(json.dumps(fittedParams, indent=2))

        # Cumulative distribution/density function
        perFeatureScores = normalizedFeatures.apply(calculate_score_for_series, args=(
            fittedParams, False), axis=0).add_suffix("_CDF")
        # perFeatureScores.head()
        # DIST = st.expon

        # def create_pdf(dist, params, size=10000):
        #    # Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    start = dist.ppf(0.01, *arg, loc=loc,
        #                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        #    end = dist.ppf(0.99999, *arg, loc=loc,
        #                   scale=scale) if arg else dist.ppf(0.99999, loc=loc, scale=scale)
        #    x = np.linspace(start, end, size)
        #    y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #    pdf = pd.Series(y, x)
        #    return pdf

        # fit exponential distribution
        # fig, axs = plt.subplots(1, 4, figsize=(
        # 15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        # axs = axs.ravel()
        # i = 0
        # for col in normalizedFeatures:
        #   print(col)
        #   feature = normalizedFeatures[col]
        #   #only fit positive values to keep the distribution tighter
        #   x = feature.values[feature.values > 0]
        #   params = DIST.fit(x)
        #   #Separate parts of parameters
        #   arg = params[:-2]
        #   loc = params[-2]
        #   scale = params[-1]
        #   #Plot
        #   pdfForPlot = create_pdf(DIST, params)
        #   pdfForPlot.plot()
        #   #Plot
        #   feature[feature.values > 0].plot(
        #   kind='hist', bins=30, )
        #   axs[i].set_ylabel('')
        #   axs[i].set_xlabel('')
        #   # Calculate SSE
        #   yhist, xhist = np.histogram(x, bins=60)
        #   xhist = (xhist + np.roll(xhist, -1))[:-1] / 2.0
        #   histPdf = DIST.pdf(xhist, loc=loc, scale=scale, *arg)
        #   sse = np.sum(np.power(yhist - histPdf, 2.0))
        #   print("sse:", sse)
        #   i += 1
        #   axs[1].set_xlabel('Events per km')
        #   axs[0].set_ylabel('Number of drivers')
        # fig.write_html('./HTML Charts/Year/exponential_curve,html')

        # Calculate driver score
        def calculate_joint_score(perFeatureScores):
            driverScores = perFeatureScores
            featureCols = [col for col in driverScores if col.startswith(
                'Hard') | col.startswith('Over')]
            driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                axis=1) / len(featureCols)))
            driverScores = driverScores.sort_values('score')
            driverScores['rank'] = len(
                driverScores['score']) - rankdata(driverScores['score']) + 1
            return driverScores

        driverScores = calculate_joint_score(perFeatureScores)
        driverScores = driverScores.reset_index()
        driverScores = driverScores.rename(columns={'DriverID': 'Driver ID', 'Hard Acceleration_CDF': 'Hard Acceleration',
                                                    'Hard Braking_CDF': 'Hard Braking',
                                           'Hard Cornering_CDF': 'Hard Cornering', 'Overspeed_CDF': 'Overspeed',
                                                    'score': 'Score', 'rank': 'Position'}, inplace=False)
        # print(driverScores.head())
        driverScores['Score'] = driverScores['Score']*100
        driverScores['Position'] = driverScores['Position']
        driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
        driverScores['Hard Braking'] = driverScores['Hard Braking']*100
        driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
        driverScores['Overspeed'] = driverScores['Overspeed']*100
        # print(driverScores)

        def condition(x):
            if x < 25:
                return "Perfectly Risky"
            elif x >= 25 and x < 50:
                return "Somewhat Risky"
            elif x >= 50 and x < 75:
                return 'Somewhat Safe'
            else:
                return 'Perfectly Safe'
        driverScores['Safety Class'] = driverScores['Score'].apply(
            condition)

        driverScores1 = driverScores[['Driver ID',
                                     'Score', 'Position', 'Safety Class']]
        driverScores1 = driverScores1.sort_values(
            by='Score', ascending=False)

        result = driverScores1.to_json(orient="records")
        # parsed = json.loads(result)
        # json.dumps(parsed, indent=4)
        # print("Score obtained per driver")
        # print(driverScores1)

        fig = px.bar(driverScores, y='Score', x='Driver ID',
                     color='Safety Class',
                     hover_data=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed', 'Position'])
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            xaxis_title="Driver ID",
            yaxis_title='Score',
            title="Bar Chart of Driver Score",
            template='plotly_white'
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Bar Chart of Driver Score.html')

        # fig.write_html('./HTML Charts/Top 10 Risky Drivers.html')
        return {"message": "Success", "status": 200, "data": result}


@app.get('/getAnnualScorePerDriver/{year}/{driverID}')
async def get_driver_score_per_year_per_driver(year: int, driverID: int):
    data = updated_data_
    day = data[(data['eventyear'] == year) & (data['DriverID'] == driverID)]

    if day.shape[0] < 600:
        return {"message": "Not Enough Records, Please Select Another Month", "status": 401}
    else:
        day = data[(data['eventyear'] == year) &
                   (data['DriverID'] == driverID)]
        # print('A preview at the selected month', day.head())
        # eventCount = len(day.event)
        # print(
        # f'The number of events made in the selected month is: {eventCount} events'
        # )
        # driverCount = day.DriverID.nunique()
        # print(
        # f"The number of drivers in this date range is: {driverCount}")
        eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            {"event": "count"}).add_prefix('Number of ')
        # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
        # print(
        # f'The average number of events made by the drivers is: {averageNoEvents}')

        fig = px.bar(eventsPerDriver.reset_index(),  x='Number of event',
                     y='DriverID', color='Number of event', barmode='group', orientation='h')
        fig.update_layout(
            yaxis_title="Number of Events",
            xaxis_title="driver ID",
            legend_title="DriverID",
            title="Bar Chart of Trips made per driver",
            template="plotly_white",
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Number of trips per Driver.html')

        fig = px.histogram(eventsPerDriver.reset_index(),
                           x='Number of event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title="Histogram of Event Counts",
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Histogram of Events.html')

        # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
        # DriverID = eventsPerDriver['Number of event'].idxmax()
        # print(
        # f'The driver with the most events is: driver {DriverID}, and the number of events made is: {maxEventsPerDriver}')
        # minEventsPerdriver = eventsPerDriver['Number of event'].min()
        # DriverID = eventsPerDriver['Number of event'].idxmin()
        # print(
        # f'The driver with the least events is: driver {DriverID}, and the number of events made is: {minEventsPerdriver}')
        # Event Type
        dfReasonHist = day.groupby(['event'])[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=True)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Count of Events",
            title='Bar Plot of Event Distribution',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Bar Plot of Event Distribution.html')

        # Handling Behavioral and Non-behavioral Events
        # non-behavioral events
        # non_behavioral_events = [event for event in day.event if event not in [
        #    'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # newCount = len(day[day['event'].isin(non_behavioral_events)])
        # print("Number of events before removing non-behavioral events is: {}.\nAfter removing non-behavioral events, we have: {} events.\nThis led to a reduction in the data size by: {:0.2f}%, leaving: {:0.2f}% of the entire data size.\nCurrent number of events is: {}".format(
        # len(day), newCount, ((len(day) - newCount)/len(day))*100, (100-(((len(day) - newCount)/len(day))*100)), newCount))
        # Specifying behavioral events
        behavioral_events = [event for event in day.event if event in [
            'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # Initializing the number of minimum events to 50 (intuitively)

        def prepData(day, minRecordsPerSubscriber=0):
            day.reset_index(inplace=True)
            # print(
            # f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
            # Remove NAs
            # df = day.dropna()
            # print(f"Removing NAs, we are left with: {len(df)} trips")
            # Filter out unwanted events
            df4 = day[day['event'].isin(behavioral_events)]
            # print(
            # f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
            # Filter out users with too few samples
            eventCountPerdriver = df4.groupby(
                'DriverID')['DriverID'].agg('count')
            driversWithManyRecords = eventCountPerdriver[eventCountPerdriver >
                                                         minRecordsPerSubscriber]
            driversWithManyRecords.keys()
            df5 = df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            # print(
            # f"Filtering users with too few samples,  we are left with: {len(df5)} trips")
            # print("*** Done. ***")
            return (df5)

        df6 = prepData(day)
        relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
            {"event": "count"}).sort_values(by='event', ascending=False)

        fig = px.histogram(relevantEventsPerSubscriber, x='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title='Histogram of Event Counts',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Histogram of Event Counts.html')

        # Distribution of Events
        dfReasonHist = df6.groupby('event')[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=False)
        # print('Distribution of Events (in descending order)')
        # print(dfReasonHist)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Event",
            title='Distribution of Behavioral Events',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Behavioral Events.html')

        # Hard Acceleration Visual
        hard_acceleration = day[day['event'] == 'hardAcceleration']
        ha = hard_acceleration[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ha = ha.reset_index()
        ha = ha.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Acceleration Events per Driver')
        # print(ha)
        # print("")

        fig = px.bar(ha, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Acceleration per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Hard Acceleration per Driver.html')

        # Hard Braking Visual
        hard_braking = day[day['event'] == 'hardBraking']
        hb = hard_braking[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hb = hb.reset_index()
        hb = hb.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Braking Events per Driver')
        # print(hb)
        # print("")

        fig = px.bar(hb, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Braking per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Hard Braking per Driver.html')

        # Hard Cornering Visual
        hard_cornering = day[day['event'] == 'hardCornering']
        hc = hard_cornering[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hc = hc.reset_index()
        hc = hc.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Cornering Events per Driver')
        # print(hc)
        # print("")

        fig = px.bar(hc, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Cornering per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Hard Cornering per Driver.html')

        # Overspeed Visual

        overspeed = day[day['event'] == 'overspeed']
        ovs = overspeed[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ovs = ovs.reset_index()
        ovs = ovs.sort_values(by='Number of event', ascending=False)
        # print('The number of Overspeeding Events per Driver')
        # print(ovs)
        # print("")

        fig = px.bar(ovs, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Overspeeding per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/Distribution of Overspeeding per Driver.html')

        # Calculate distance traveled in each trip using Haversine

        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        def total_distance(oneDriver):
            dist = haversine(oneDriver.longitude.shift(1), oneDriver.latitude.shift(1),
                             oneDriver.loc[1:, 'longitude'], oneDriver.loc[1:, 'latitude'])
            return np.sum(dist)

        # Calculate the overall distance made per driver

        def calculate_overall_distance_traveled(dfRaw):
            dfDistancePerdriver = day.groupby('DriverID').apply(
                total_distance).reset_index(name='Distance')
            return dfDistancePerdriver
        distancePerdriver = calculate_overall_distance_traveled(df6)
        # print('Distance Traveled per Driver (in descending order)')
        # print(distancePerdriver.sort_values(by='Distance', ascending=False))

        fig = px.bar(distancePerdriver.sort_values(by='Distance', ascending=True),
                     x='DriverID', y='Distance', color='Distance', )
        fig.update_layout(
            xaxis_title="Driver ID",
            yaxis_title="Distance Traveled",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            title='Distance Traveled per Driver',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Distance Traveled per Driver.html')

        # Feature Engineering
        # Transform the events data frame to a features data frame (column for each type of relevant event)

        def create_feature_set(df6, distancePerdriver):
            dfEventAggByDriver = df6.groupby(['DriverID', 'event'])[['event']].agg(
                'count').add_prefix('Number of ').reset_index()
            # Pivot events into columns
            # Pivot the table by setting the drivers' name as the index column, while the respective events takes on a column each.
            # Finally, fill missing observations with zeros(0s)
            dfEventMatrix = dfEventAggByDriver.pivot(index='DriverID', columns=[
                                                     'event'], values='Number of event').add_prefix('F_').fillna(0).reset_index()
            # Merge the created pivot table with the earlier created dataframe for distance traveled per driver.
            dfEventMatrix = dfEventMatrix.merge(
                distancePerdriver, how='inner', on='DriverID')
            dfEventMatrix.set_index('DriverID', inplace=True)
            # Set the columns to start with F_
            featureCols = [
                col for col in dfEventMatrix if col.startswith('F_')]
            # Divide each of the features by the traveled.
            dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                dfEventMatrix['Distance'], axis=0)
            dfFeatureSet = dfEventMatrix[featureCols]
            return dfFeatureSet

        features = create_feature_set(df6, distancePerdriver)

        # Motion based events
        try:
            if 'F_hardAcceleration' in features:
                features['F_hardAcceleration'] = features['F_hardAcceleration']
            else:
                features['F_hardAcceleration'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardBraking' in features:
                features['F_hardBraking'] = features['F_hardBraking']
            else:
                features['F_hardBraking'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardCornering' in features:
                features['F_hardCornering'] = features['F_hardCornering']
            else:
                features['F_hardCornering'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_overspeed' in features:
                features['F_overspeed'] = features['F_overspeed']
            else:
                features['F_overspeed'] = pd.Series(
                    [0 for x in range(len(features.index))])
            # features = [['F_hardAcceleration', 'F_hardBraking', 'F_hardCornering', 'F_overspeed']]
            features = features.fillna(0)
            # print(features)
        except Exception as e:
            print('Error')
            logger.warning(e)
            logger.error('Exception occurred', exc_info=True)

        features = features.rename(columns={'F_hardAcceleration': "Hard Acceleration",
                                            'F_hardBraking': "Hard Braking",
                                            'F_hardCornering': "Hard Cornering",
                                            'F_overspeed': 'Overspeed'}, inplace=False)
        # print(features.head())

        # Driver with the lowest harsh braking score
        # print(
        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmin()])

        # Driver with the highest harsh braking score
        # print(

        #    f'The information about the driver with the least hard braking score is given below: ',

        #    features.loc[features['Hard Acceleration'].idxmax()])

        fig = px.scatter_matrix(features.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()  # type: ignore
        fig.write_html('./HTML Charts/Year/scatter_matrix_features.html')

        # Detecting and Handling Outliers
        # Log Transformation
        features_log = np.log1p(features)
        features_log = features.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        features_log = features.fillna(0)
        # print('A preview of the data upon the application of Log Transformation')
        # print(features_log.head())

        fig = px.scatter_matrix(features_log.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/scatter_matrix_log_transformation.html')

        # Box-Cox Transformation
        # print('Applying Box-Cox transformation on the data')

        def transform_to_normal(x, min_max_transform=False):
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            valueGreaterThanZero = np.where(x <= 0, 0, x)
            positives = x[valueGreaterThanZero == 1]
            if (len(positives) > 0):
                xt[valueGreaterThanZero == 1], _ = st.boxcox(positives+1)
            if min_max_transform:
                xt = (xt - np.min(xt)) / (np.max(xt)-np.min(xt))
            return xt
        # features_1 = features_1.set_index('DriverId')
        transFeatures = features.apply(lambda x: (
            transform_to_normal(x, min_max_transform=True)))
        transFeatures = transFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeatures = transFeatures.fillna(0)
        # print('A preview of the data upon the application of Box-Cox Transformation')
        # print(transFeatures.head())

        fig = px.scatter_matrix(transFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking',
                                                                         'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html('./HTML Charts/Year/scatter_matrix_box_cox.html')

        # Standard Deviation Rule
        # print('Checking for and replacement of outliers')

        def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
            print(x.name)
            x = x.values
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            xt = transform_to_normal(x)
            xMean, xStd = np.mean(xt), np.std(xt)
            outliers = np.where(xt > xMean + stdFactor*xStd)[0]
            inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
            if len(outliers) > 0:
                # print("found outlier with factor: " +
                # str(stdFactor)+" : "+str(outliers))
                xinline = x[inliers]
                maxInRange = np.max(xinline)
                # print("replacing outliers {} with max={}".format(
                # outliers, maxInRange))
                vals = x.copy()
                vals[outliers] = maxInRange
                x = pd.Series(vals)
            else:
                pass
            if normalize:
                # Normalize to [0,1]
                x = (x - np.min(x)) / (np.max(x)-np.min(x))
            return x
        # features_1 = features_1.reset_index()
        cleanFeatures = features.apply(
            lambda x: (replace_outliers_with_limit(x)))
        cleanFeatures = cleanFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        cleanFeatures = cleanFeatures.fillna(0)
        # print("A preview of the cleaned data after handling outliers")
        # print(cleanFeatures.head())

        fig = px.scatter_matrix(cleanFeatures.reset_index(), dimensions=['Hard Acceleration',
                                                                         'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year/scatter_matrix_cleaned_features.html')

        # Correlation between events
        # print("Correlation between the events")
        corr = cleanFeatures.corr()
        corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
        corr = corr.fillna(0)
        # print(corr)

        # Plot the heatmap of the correlation matrix
        # print(f"""
        #               ### Correlation Heatmap
        #                """)
        fig = px.imshow(corr, color_continuous_scale='hot',
                        title='Correlation Heatmap', width=600, height=500, aspect='equal')
        # fig.show()
        fig.write_html('./HTML Charts/Year/correlation heatmap.html')

        # Pre step: Normalize features
        # print("Data normalization")
        minPerFeature = cleanFeatures.min()
        maxPerFeature = cleanFeatures.max()
        # print("Min and Max values per column before normalization")
        # for col in range(0, len(cleanFeatures.columns)):
        # print(
        # f"{cleanFeatures.columns[col]} range:[{minPerFeature[col]},{maxPerFeature[col]}]")
        normalizedFeatures = (cleanFeatures-minPerFeature) / \
            (maxPerFeature-minPerFeature)
        normalizedFeatures = normalizedFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        normalizedFeatures = normalizedFeatures.fillna(0)
        # print("A preview of the normalized data")
        # print(normalizedFeatures.head())

        # Standardize features after box-cox as well.
        # print("Standardizing the features after Box-Cox Transformation")
        transFeaturesScaled = (
            transFeatures - transFeatures.mean())/transFeatures.std()
        transFeaturesScaled = transFeaturesScaled.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeaturesScaled = transFeaturesScaled.fillna(0)
        # print("A preview of the standardized data")
        # print(transFeaturesScaled.head())

        # print("Mean and STD before standardization")
        # for col in range(0, len(transFeatures.columns)):
        # print(
        # f"{transFeatures.columns[col]} range:[{transFeatures.mean()[col]},{transFeatures.std()[col]}]")

        # Anomaly Detection:
        # LOF - Local Outlier Filter
        # X = transFeaturesScaled.values
        # X = np.nan_to_num(X)
        # clf = LocalOutlierFactor(n_neighbors=5)
        # isOutlier = clf.fit_predict(X)
        # plt.title("Local Outlier Factor (LOF)", fontsize=20)
        # a = plt.scatter(X[isOutlier == 1, 0], X[isOutlier == 1, 1], c='white',
        #                edgecolor='k', s=40)
        # b = plt.scatter(X[isOutlier == -1, 0], X[isOutlier == -1, 1], c='red',
        #                edgecolor='k', s=40)
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # plt.xlabel(normalizedFeatures.columns[0], fontsize=15)
        # plt.ylabel(normalizedFeatures.columns[1], fontsize=15)
        # plt.xlim((-0.01, 1.01))
        # plt.ylim((-0.01, 1.01))
        # plt.legend([a, b],
        #           ["normal observations",
        #            "abnormal observations"],
        #           loc="upper right", prop={'size': 15}, frameon=True)
        # fig.show()
        # fig.write_html('./HTML Charts/Year/lof.html')

        # Multivariate analysis
        # Dimensionality reduction
        # PCA
        # pca = PCA(n_components=4)
        # principalComponents = pca.fit_transform(normalizedFeatures)
        # column_names = ['principal component {}'.format(
        #    i) for i in range(normalizedFeatures.shape[1])]
        # plt.bar(x=column_names, height=pca.explained_variance_ratio_)
        # plt.title("Percentage of explained variance")
        # fig.show()
        # print("Principal components explained variance ratio: {}.".format(
        # pca.explained_variance_ratio_))
        # principalDf = pd.DataFrame(
        #    data=principalComponents, columns=column_names)
        # df = normalizedFeatures
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # Show correlation matrix of the original features and the first principal component
        # pcAndOriginal = pd.concat(
        #    [principalDf.iloc[:, 0].reset_index(drop=True), normalizedFeatures], axis=1)
        # sns.set(style="ticks")
        # histplot = pcAndOriginal['principal component 0'].hist(figsize=(5, 5))
        # histplot.set_title("principal component 0 histogram")
        # sns.pairplot(pcAndOriginal, y_vars=['principal component 0'],
        #             x_vars=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'])
        # fig.write_html('./HTML Charts/Year/correlation_matrix_pca.html')

        # Extract statistics from the fitted distributions
        # print(normalizedFeatures.head())

        # Fit exponential distribution
        def fit_distribution_params(series):
            # print("Extracting distribution parameters for feature: " +
            #      series.name + " (" + str(len(series)) + ' values)')
            xPositive = series[series > 0]
            xPositive = xPositive.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            xPositive = xPositive.fillna(0)
            # xPositive = xPositive.replace
            probs = np.zeros(len(series))
            if (len(xPositive) > 0):
                params = st.expon.fit(xPositive)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # print('params = {}, {}, {}.'.format(arg, loc, scale))
                return arg, loc, scale

        def calculate_score_for_series(x, fittedParams, verbose=False):
            # print("Calculating scores for feature: " + x.name)
            xPositive = x[x > 0]
            probs = np.zeros(len(x))
            if (len(xPositive) > 0):
                arg = fittedParams[x.name]['arg']
                loc = fittedParams[x.name]['loc']
                scale = fittedParams[x.name]['scale']
                probs[x > 0] = st.expon.cdf(
                    xPositive, loc=loc, scale=scale, *arg)
                if verbose:
                    probs_df = pd.DataFrame(
                        {'Event value': x.values.tolist(), 'Event probability': probs}, index=True)
                    probs_df = probs_df.sort_values(by='Event value')
                    # print(probs_df)
            return probs

        # Store each fitted distribution parameters for later use
        fittedParams = {}
        for col in features.columns:
            arg, loc, scale = fit_distribution_params(features[col])
            fittedParams[col] = {}
            fittedParams[col]['arg'] = arg
            fittedParams[col]['loc'] = loc
            fittedParams[col]['scale'] = scale
        # print('Fitted parameters:')
        # print(json.dumps(fittedParams, indent=2))

        # Cumulative distribution/density function
        perFeatureScores = normalizedFeatures.apply(calculate_score_for_series, args=(
            fittedParams, False), axis=0).add_suffix("_CDF")
        # perFeatureScores.head()
        # DIST = st.expon

        # def create_pdf(dist, params, size=10000):
        #    # Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    start = dist.ppf(0.01, *arg, loc=loc,
        #                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        #    end = dist.ppf(0.99999, *arg, loc=loc,
        #                   scale=scale) if arg else dist.ppf(0.99999, loc=loc, scale=scale)
        #    x = np.linspace(start, end, size)
        #    y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #    pdf = pd.Series(y, x)
        #    return pdf

        # fit exponential distribution
        # fig, axs = plt.subplots(1, 4, figsize=(
        # 15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        # axs = axs.ravel()
        # i = 0
        # for col in normalizedFeatures:
        #   print(col)
        #   feature = normalizedFeatures[col]
        #   #only fit positive values to keep the distribution tighter
        #   x = feature.values[feature.values > 0]
        #   params = DIST.fit(x)
        #   #Separate parts of parameters
        #   arg = params[:-2]
        #   loc = params[-2]
        #   scale = params[-1]
        #   #Plot
        #   pdfForPlot = create_pdf(DIST, params)
        #   pdfForPlot.plot()
        #   #Plot
        #   feature[feature.values > 0].plot(
        #   kind='hist', bins=30, )
        #   axs[i].set_ylabel('')
        #   axs[i].set_xlabel('')
        #   # Calculate SSE
        #   yhist, xhist = np.histogram(x, bins=60)
        #   xhist = (xhist + np.roll(xhist, -1))[:-1] / 2.0
        #   histPdf = DIST.pdf(xhist, loc=loc, scale=scale, *arg)
        #   sse = np.sum(np.power(yhist - histPdf, 2.0))
        #   print("sse:", sse)
        #   i += 1
        #   axs[1].set_xlabel('Events per km')
        #   axs[0].set_ylabel('Number of drivers')
        # fig.write_html('./HTML Charts/Year/exponential_curve,html')

        # Calculate driver score
        def calculate_joint_score(perFeatureScores):
            driverScores = perFeatureScores
            featureCols = [col for col in driverScores if col.startswith(
                'Hard') | col.startswith('Over')]
            driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                axis=1) / len(featureCols)))
            driverScores = driverScores.sort_values('score')
            driverScores['rank'] = len(
                driverScores['score']) - rankdata(driverScores['score']) + 1
            return driverScores

        driverScores = calculate_joint_score(perFeatureScores)
        driverScores = driverScores.reset_index()
        driverScores = driverScores.rename(columns={'DriverID': 'Driver ID', 'Hard Acceleration_CDF': 'Hard Acceleration',
                                                    'Hard Braking_CDF': 'Hard Braking',
                                           'Hard Cornering_CDF': 'Hard Cornering', 'Overspeed_CDF': 'Overspeed',
                                                    'score': 'Score', 'rank': 'Position'}, inplace=False)
        # print(driverScores.head())
        driverScores['Score'] = driverScores['Score']*100
        driverScores['Position'] = driverScores['Position']
        driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
        driverScores['Hard Braking'] = driverScores['Hard Braking']*100
        driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
        driverScores['Overspeed'] = driverScores['Overspeed']*100
        # print(driverScores)

        def condition(x):
            if x < 25:
                return "Perfectly Risky"
            elif x >= 25 and x < 50:
                return "Somewhat Risky"
            elif x >= 50 and x < 75:
                return 'Somewhat Safe'
            else:
                return 'Perfectly Safe'
        driverScores['Safety Class'] = driverScores['Score'].apply(
            condition)

        driverScores1 = driverScores[['Driver ID',
                                     'Score', 'Position', 'Safety Class']]
        driverScores1 = driverScores1.sort_values(
            by='Score', ascending=False)

        result = driverScores1.to_json(orient="records")
        # parsed = json.loads(result)
        # json.dumps(parsed, indent=4)
        # print("Score obtained per driver")
        # print(driverScores1)

        fig = px.bar(driverScores, y='Score', x='Driver ID',
                     color='Safety Class',
                     hover_data=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed', 'Position'])
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            xaxis_title="Driver ID",
            yaxis_title='Score',
            title="Bar Chart of Driver Score",
            template='plotly_white'
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year/Bar Chart of Driver Score.html')

        # fig.write_html('./HTML Charts/Top 10 Risky Drivers.html')
        return {"message": "Success", "status": 200, "data": result}


@app.get('/getAnnualScorePerMonth/{year}/{month}')
async def get_driver_score_per_year_per_month(year: int, month: str):
    data = updated_data_
    day = data[(data['eventyear'] == year) & (data['eventmonth'] == month)]

    if day.shape[0] < 600:
        return {"message": "Not Enough Records, Please Select Another Month", "status": 401}
    else:
        day = data[(data['eventyear'] == year) & (data['eventmonth'] == month)]
        # print('A preview at the selected month', day.head())
        # eventCount = len(day.event)
        # print(
        # f'The number of events made in the selected month is: {eventCount} events'
        # )
        # driverCount = day.DriverID.nunique()
        # print(
        # f"The number of drivers in this date range is: {driverCount}")
        eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            {"event": "count"}).add_prefix('Number of ')
        # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
        # print(
        # f'The average number of events made by the drivers is: {averageNoEvents}')

        fig = px.bar(eventsPerDriver.reset_index(),  x='Number of event',
                     y='DriverID', color='Number of event', barmode='group', orientation='h')
        fig.update_layout(
            yaxis_title="Number of Events",
            xaxis_title="driver ID",
            legend_title="DriverID",
            title="Bar Chart of Trips made per Driver",
            template="plotly_white",
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Bar Chart of Trips made per Driver.html')

        fig = px.histogram(eventsPerDriver.reset_index(),
                           x='Number of event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title="Histogram of Event Counts",
            template="plotly_white"
        )
        # fig.show()
        fig.write_html('./HTML Charts/Year and Month/Histogram of Event.html')

        # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
        # DriverID = eventsPerDriver['Number of event'].idxmax()
        # print(
        # f'The driver with the most events is: driver {DriverID}, and the number of events made is: {maxEventsPerDriver}')
        # minEventsPerdriver = eventsPerDriver['Number of event'].min()
        # DriverID = eventsPerDriver['Number of event'].idxmin()
        # print(
        # f'The driver with the least events is: driver {DriverID}, and the number of events made is: {minEventsPerdriver}')
        # Event Type
        dfReasonHist = day.groupby(['event'])[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=True)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Count of Events",
            title='Bar Plot of Event Distribution',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Bar Plot of Event Distribution.html')

        # Handling Behavioral and Non-behavioral Events
        # non-behavioral events
        # non_behavioral_events = [event for event in day.event if event not in [
        #    'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # newCount = len(day[day['event'].isin(non_behavioral_events)])
        # print("Number of events before removing non-behavioral events is: {}.\nAfter removing non-behavioral events, we have: {} events.\nThis led to a reduction in the data size by: {:0.2f}%, leaving: {:0.2f}% of the entire data size.\nCurrent number of events is: {}".format(
        # len(day), newCount, ((len(day) - newCount)/len(day))*100, (100-(((len(day) - newCount)/len(day))*100)), newCount))
        # Specifying behavioral events
        behavioral_events = [event for event in day.event if event in [
            'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # Initializing the number of minimum events to 50 (intuitively)

        def prepData(day, minRecordsPerSubscriber=0):
            day.reset_index(inplace=True)
            # print(
            # f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
            # Remove NAs
            # df = day.dropna()
            # print(f"Removing NAs, we are left with: {len(df)} trips")
            # Filter out unwanted events
            df4 = day[day['event'].isin(behavioral_events)]
            # print(
            # f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
            # Filter out users with too few samples
            eventCountPerdriver = df4.groupby(
                'DriverID')['DriverID'].agg('count')
            driversWithManyRecords = eventCountPerdriver[eventCountPerdriver >
                                                         minRecordsPerSubscriber]
            driversWithManyRecords.keys()
            df5 = df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            # print(
            # f"Filtering users with too few samples,  we are left with: {len(df5)} trips")
            # print("*** Done. ***")
            return (df5)

        df6 = prepData(day)
        relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
            {"event": "count"}).sort_values(by='event', ascending=False)

        fig = px.histogram(relevantEventsPerSubscriber, x='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title='Histogram of Event Counts',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Histogram of Event Counts.html')

        # Distribution of Events
        dfReasonHist = df6.groupby('event')[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=False)
        # print('Distribution of Events (in descending order)')
        # print(dfReasonHist)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Event",
            title='Distribution of Behavioral Events',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Distribution of Behavioral Events.html')

        # Hard Acceleration Visual
        hard_acceleration = day[day['event'] == 'hardAcceleration']
        ha = hard_acceleration[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ha = ha.reset_index()
        ha = ha.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Acceleration Events per Driver')
        # print(ha)
        # print("")

        fig = px.bar(ha, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Acceleration per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Distribution of Hard Acceleration per Driver.html')

        # Hard Braking Visual
        hard_braking = day[day['event'] == 'hardBraking']
        hb = hard_braking[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hb = hb.reset_index()
        hb = hb.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Braking Events per Driver')
        # print(hb)
        # print("")

        fig = px.bar(hb, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Braking per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Distribution of Hard Braking per Driver.html')

        # Hard Cornering Visual
        hard_cornering = day[day['event'] == 'hardCornering']
        hc = hard_cornering[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hc = hc.reset_index()
        hc = hc.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Cornering Events per Driver')
        # print(hc)
        # print("")

        fig = px.bar(hc, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Cornering per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Distribution of Hard Cornering per Driver.html')

        # Overspeed Visual

        overspeed = day[day['event'] == 'overspeed']
        ovs = overspeed[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ovs = ovs.reset_index()
        ovs = ovs.sort_values(by='Number of event', ascending=False)
        # print('The number of Overspeeding Events per Driver')
        # print(ovs)
        # print("")

        fig = px.bar(ovs, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Overspeeding per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Distribution of Overspeeding per Driver.html')

        # Calculate distance traveled in each trip using Haversine

        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        def total_distance(oneDriver):
            dist = haversine(oneDriver.longitude.shift(1), oneDriver.latitude.shift(1),
                             oneDriver.loc[1:, 'longitude'], oneDriver.loc[1:, 'latitude'])
            return np.sum(dist)

        # Calculate the overall distance made per driver

        def calculate_overall_distance_traveled(dfRaw):
            dfDistancePerdriver = day.groupby('DriverID').apply(
                total_distance).reset_index(name='Distance')
            return dfDistancePerdriver
        distancePerdriver = calculate_overall_distance_traveled(df6)
        # print('Distance Traveled per Driver (in descending order)')
        # print(distancePerdriver.sort_values(by='Distance', ascending=False))

        fig = px.bar(distancePerdriver.sort_values(by='Distance', ascending=True),
                     x='DriverID', y='Distance', color='Distance', )
        fig.update_layout(
            xaxis_title="Driver ID",
            yaxis_title="Distance Traveled",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            title='Distance Traveled per Driver',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Distance Traveled per Driver.html')
        # Feature Engineering
        # Transform the events data frame to a features data frame (column for each type of relevant event)

        def create_feature_set(df6, distancePerdriver):
            dfEventAggByDriver = df6.groupby(['DriverID', 'event'])[['event']].agg(
                'count').add_prefix('Number of ').reset_index()
            # Pivot events into columns
            # Pivot the table by setting the drivers' name as the index column, while the respective events takes on a column each. Finally, fill missing observations with zeros(0s)
            dfEventMatrix = dfEventAggByDriver.pivot(index='DriverID', columns=[
                                                     'event'], values='Number of event').add_prefix('F_').fillna(0).reset_index()
            # Merge the created pivot table with the earlier created dataframe for distance traveled per driver.
            dfEventMatrix = dfEventMatrix.merge(
                distancePerdriver, how='inner', on='DriverID')
            dfEventMatrix.set_index('DriverID', inplace=True)
            # Set the columns to start with F_
            featureCols = [
                col for col in dfEventMatrix if col.startswith('F_')]
            # Divide each of the features by the traveled.
            dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                dfEventMatrix['Distance'], axis=0)
            dfFeatureSet = dfEventMatrix[featureCols]
            return dfFeatureSet

        features = create_feature_set(df6, distancePerdriver)

        # Motion based events
        try:
            if 'F_hardAcceleration' in features:
                features['F_hardAcceleration'] = features['F_hardAcceleration']
            else:
                features['F_hardAcceleration'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardBraking' in features:
                features['F_hardBraking'] = features['F_hardBraking']
            else:
                features['F_hardBraking'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardCornering' in features:
                features['F_hardCornering'] = features['F_hardCornering']
            else:
                features['F_hardCornering'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_overspeed' in features:
                features['F_overspeed'] = features['F_overspeed']
            else:
                features['F_overspeed'] = pd.Series(
                    [0 for x in range(len(features.index))])
            # features = [['F_hardAcceleration', 'F_hardBraking', 'F_hardCornering', 'F_overspeed']]
            features = features.fillna(0)
            # print(features)
        except Exception as e:
            print('Error')
            logger.warning(e)
            logger.error('Exception occurred', exc_info=True)

        features = features.rename(columns={'F_hardAcceleration': "Hard Acceleration",
                                            'F_hardBraking': "Hard Braking",
                                            'F_hardCornering': "Hard Cornering",
                                            'F_overspeed': 'Overspeed'}, inplace=False)
        # print(features.head())

        # Driver with the lowest harsh braking score
        # print(
        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmin()])

        # Driver with the highest harsh braking score
        # print(

        #    f'The information about the driver with the least hard braking score is given below: ',

        #    features.loc[features['Hard Acceleration'].idxmax()])

        fig = px.scatter_matrix(features.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()  # type: ignore
        fig.write_html(
            './HTML Charts/Year and Month/scatter_matrix_features.html')

        # Detecting and Handling Outliers
        # Log Transformation
        features_log = np.log1p(features)
        features_log = features.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        features_log = features.fillna(0)
        # print('A preview of the data upon the application of Log Transformation')
        # print(features_log.head())

        fig = px.scatter_matrix(features_log.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/scatter_matrix_features_log.html')

        # Box-Cox Transformation
        # print('Applying Box-Cox transformation on the data')

        def transform_to_normal(x, min_max_transform=False):
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            valueGreaterThanZero = np.where(x <= 0, 0, x)
            positives = x[valueGreaterThanZero == 1]
            if (len(positives) > 0):
                xt[valueGreaterThanZero == 1], _ = st.boxcox(positives+1)
            if min_max_transform:
                xt = (xt - np.min(xt)) / (np.max(xt)-np.min(xt))
            return xt
        # features_1 = features_1.set_index('DriverId')
        transFeatures = features.apply(lambda x: (
            transform_to_normal(x, min_max_transform=True)))
        transFeatures = transFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeatures = transFeatures.fillna(0)
        # print('A preview of the data upon the application of Box-Cox Transformation')
        # print(transFeatures.head())

        fig = px.scatter_matrix(transFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/scatter_matrix_box_cox.html')

        # Standard Deviation Rule
        # print('Checking for and replacement of outliers')

        def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
            print(x.name)
            x = x.values
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            xt = transform_to_normal(x)
            xMean, xStd = np.mean(xt), np.std(xt)
            outliers = np.where(xt > xMean + stdFactor*xStd)[0]
            inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
            if len(outliers) > 0:
                # print("found outlier with factor: " +
                # str(stdFactor)+" : "+str(outliers))
                xinline = x[inliers]
                maxInRange = np.max(xinline)
                # print("replacing outliers {} with max={}".format(
                # outliers, maxInRange))
                vals = x.copy()
                vals[outliers] = maxInRange
                x = pd.Series(vals)
            else:
                pass
            if normalize:
                # Normalize to [0,1]
                x = (x - np.min(x)) / (np.max(x)-np.min(x))
            return x
        # features_1 = features_1.reset_index()
        cleanFeatures = features.apply(
            lambda x: (replace_outliers_with_limit(x)))
        cleanFeatures = cleanFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        cleanFeatures = cleanFeatures.fillna(0)
        # print("A preview of the cleaned data after handling outliers")
        # print(cleanFeatures.head())

        fig = px.scatter_matrix(cleanFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/scatter_matrix_cleaned_features.html')

        # Correlation between events
        # print("Correlation between the events")
        corr = cleanFeatures.corr()
        corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
        corr = corr.fillna(0)
        # print(corr)

        # Plot the heatmap of the correlation matrix
        # print(f"""
        #               ### Correlation Heatmap
        #                """)
        fig = px.imshow(corr, color_continuous_scale='hot',
                        title='Correlation Heatmap', width=600, height=500, aspect='equal')
        # fig.show()
        fig.write_html('./HTML Charts/Year and Month/Correlation Heatmap.html')

        # Pre step: Normalize features
        # print("Data normalization")
        minPerFeature = cleanFeatures.min()
        maxPerFeature = cleanFeatures.max()
        # print("Min and Max values per column before normalization")
        # for col in range(0, len(cleanFeatures.columns)):
        # print(
        # f"{cleanFeatures.columns[col]} range:[{minPerFeature[col]},{maxPerFeature[col]}]")
        normalizedFeatures = (cleanFeatures-minPerFeature) / \
            (maxPerFeature-minPerFeature)
        normalizedFeatures = normalizedFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        normalizedFeatures = normalizedFeatures.fillna(0)
        # print("A preview of the normalized data")
        # print(normalizedFeatures.head())

        # Standardize features after box-cox as well.
        # print("Standardizing the features after Box-Cox Transformation")
        transFeaturesScaled = (
            transFeatures - transFeatures.mean())/transFeatures.std()
        transFeaturesScaled = transFeaturesScaled.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeaturesScaled = transFeaturesScaled.fillna(0)
        # print("A preview of the standardized data")
        # print(transFeaturesScaled.head())

        # print("Mean and STD before standardization")
        # for col in range(0, len(transFeatures.columns)):
        # print(
        # f"{transFeatures.columns[col]} range:[{transFeatures.mean()[col]},{transFeatures.std()[col]}]")

        # Anomaly Detection:
        # LOF - Local Outlier Filter
        # X = transFeaturesScaled.values
        # X = np.nan_to_num(X)
        # clf = LocalOutlierFactor(n_neighbors=5)
        # isOutlier = clf.fit_predict(X)
        # plt.title("Local Outlier Factor (LOF)", fontsize=20)
        # a = plt.scatter(X[isOutlier == 1, 0], X[isOutlier == 1, 1], c='white',
        #                edgecolor='k', s=40)
        # b = plt.scatter(X[isOutlier == -1, 0], X[isOutlier == -1, 1], c='red',
        #                edgecolor='k', s=40)
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # plt.xlabel(normalizedFeatures.columns[0], fontsize=15)
        # plt.ylabel(normalizedFeatures.columns[1], fontsize=15)
        # plt.xlim((-0.01, 1.01))
        # plt.ylim((-0.01, 1.01))
        # plt.legend([a, b],
        #           ["normal observations",
        #            "abnormal observations"],
        #           loc="upper right", prop={'size': 15}, frameon=True)
        # fig.show()
        # fig.write_html('./HTML Charts/Year and Month/lof.html')

        # Multivariate analysis
        # Dimensionality reduction
        # PCA
        # pca = PCA(n_components=4)
        # principalComponents = pca.fit_transform(normalizedFeatures)
        # column_names = ['principal component {}'.format(
        #    i) for i in range(normalizedFeatures.shape[1])]
        # plt.bar(x=column_names, height=pca.explained_variance_ratio_)
        # plt.title("Percentage of explained variance")
        # fig.show()
        # print("Principal components explained variance ratio: {}.".format(
        # pca.explained_variance_ratio_))
        # principalDf = pd.DataFrame(
        #    data=principalComponents, columns=column_names)
        # df = normalizedFeatures

        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # Show correlation matrix of the original features and the first principal component
        # pcAndOriginal = pd.concat(
        #    [principalDf.iloc[:, 0].reset_index(drop=True), normalizedFeatures], axis=1)
        # sns.set(style="ticks")
        # histplot = pcAndOriginal['principal component 0'].hist(figsize=(5, 5))
        # histplot.set_title("principal component 0 histogram")
        # sns.pairplot(pcAndOriginal, y_vars=['principal component 0'],
        #             x_vars=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'])
        # fig.write_html('./HTML Charts/Year and Month/pca.html')

        # Extract statistics from the fitted distributions
        # normalizedFeatures.head()

        # Fit exponential distribution
        def fit_distribution_params(series):
            # print("Extracting distribution parameters for feature: " +
            #      series.name + " (" + str(len(series)) + ' values)')
            xPositive = series[series > 0]
            xPositive = xPositive.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            xPositive = xPositive.fillna(0)
            # xPositive = xPositive.replace
            probs = np.zeros(len(series))
            if (len(xPositive) > 0):
                params = st.expon.fit(xPositive)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # print('params = {}, {}, {}.'.format(arg, loc, scale))
                return arg, loc, scale

        def calculate_score_for_series(x, fittedParams, verbose=False):
            # print("Calculating scores for feature: " + x.name)
            xPositive = x[x > 0]
            probs = np.zeros(len(x))
            if (len(xPositive) > 0):
                arg = fittedParams[x.name]['arg']
                loc = fittedParams[x.name]['loc']
                scale = fittedParams[x.name]['scale']
                probs[x > 0] = st.expon.cdf(
                    xPositive, loc=loc, scale=scale, *arg)
                if verbose:
                    probs_df = pd.DataFrame(
                        {'Event value': x.values.tolist(), 'Event probability': probs}, index=True)
                    probs_df = probs_df.sort_values(by='Event value')
                    # print(probs_df)
            return probs

        # Store each fitted distribution parameters for later use
        fittedParams = {}
        for col in features.columns:
            arg, loc, scale = fit_distribution_params(features[col])
            fittedParams[col] = {}
            fittedParams[col]['arg'] = arg
            fittedParams[col]['loc'] = loc
            fittedParams[col]['scale'] = scale
        # print('Fitted parameters:')
        # print(json.dumps(fittedParams, indent=2))

        # Cumulative distribution/density function
        perFeatureScores = normalizedFeatures.apply(calculate_score_for_series, args=(
            fittedParams, False), axis=0).add_suffix("_CDF")
        # perFeatureScores.head()
        # DIST = st.expon

        # def create_pdf(dist, params, size=10000):
        #    # Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    start = dist.ppf(0.01, *arg, loc=loc,
        #                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        #    end = dist.ppf(0.99999, *arg, loc=loc,
        #                   scale=scale) if arg else dist.ppf(0.99999, loc=loc, scale=scale)
        #    x = np.linspace(start, end, size)
        #    y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #    pdf = pd.Series(y, x)
        #    return pdf

        # fit exponential distribution
        # fig, axs = plt.subplots(1, 4, figsize=(
        # 15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        # axs = axs.ravel()
        # i = 0
        # for col in normalizedFeatures:
        #   print(col)
        #   feature = normalizedFeatures[col]
        #   #only fit positive values to keep the distribution tighter
        #   x = feature.values[feature.values > 0]
        #   params = DIST.fit(x)
        #   #Separate parts of parameters
        #   arg = params[:-2]
        #   loc = params[-2]
        #   scale = params[-1]
        #   #Plot
        #   pdfForPlot = create_pdf(DIST, params)
        #   pdfForPlot.plot()
        #   #Plot
        #   feature[feature.values > 0].plot(
        #   kind='hist', bins=30, )
        #   axs[i].set_ylabel('')
        #   axs[i].set_xlabel('')
        #   # Calculate SSE
        #   yhist, xhist = np.histogram(x, bins=60)
        #   xhist = (xhist + np.roll(xhist, -1))[:-1] / 2.0
        #   histPdf = DIST.pdf(xhist, loc=loc, scale=scale, *arg)
        #   sse = np.sum(np.power(yhist - histPdf, 2.0))
        #   print("sse:", sse)
        #   i += 1
        #   axs[1].set_xlabel('Events per km')
        #   axs[0].set_ylabel('Number of drivers')
        # fig.write_html('./HTML Charts/Year/exponential_curve.html')

        # Calculate driver score
        def calculate_joint_score(perFeatureScores):
            driverScores = perFeatureScores
            featureCols = [col for col in driverScores if col.startswith(
                'Hard') | col.startswith('Over')]
            driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                axis=1) / len(featureCols)))
            driverScores = driverScores.sort_values('score')
            driverScores['rank'] = len(
                driverScores['score']) - rankdata(driverScores['score']) + 1
            return driverScores

        driverScores = calculate_joint_score(perFeatureScores)
        driverScores = driverScores.reset_index()
        driverScores = driverScores.rename(columns={'DriverID': 'Driver ID', 'Hard Acceleration_CDF': 'Hard Acceleration', 'Hard Braking_CDF': 'Hard Braking',
                                           'Hard Cornering_CDF': 'Hard Cornering', 'Overspeed_CDF': 'Overspeed', 'score': 'Score', 'rank': 'Position'}, inplace=False)
        # print(driverScores.head())
        driverScores['Score'] = driverScores['Score']*100
        driverScores['Position'] = driverScores['Position']
        driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
        driverScores['Hard Braking'] = driverScores['Hard Braking']*100
        driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
        driverScores['Overspeed'] = driverScores['Overspeed']*100
        # print(driverScores)

        def condition(x):
            if x < 25:
                return "Perfectly Risky"
            elif x >= 25 and x < 50:
                return "Somewhat Risky"
            elif x >= 50 and x < 75:
                return 'Somewhat Safe'
            else:
                return 'Perfectly Safe'
        driverScores['Safety Class'] = driverScores['Score'].apply(
            condition)

        driverScores1 = driverScores[['Driver ID',
                                     'Score', 'Position', 'Safety Class']]
        driverScores1 = driverScores1.sort_values(
            by='Score', ascending=False)

        result = driverScores1.to_json(orient="records")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        # print("Score obtained per driver")
        # print(driverScores1)

        fig = px.bar(driverScores, y='Score', x='Driver ID',
                     color='Safety Class',
                     hover_data=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed', 'Position'])
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            xaxis_title="Driver ID",
            yaxis_title='Score',
            title="Bar Chart of Driver Score",
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month/Bar Chart of Driver Scores.html')
        return {"message": "Success", "status": 200, "data": result}


@app.get('/getAnnualScorePerMonthPerDriver/{year}/{month}/{id}')
async def get_driver_score_per_driver_per_year_per_month(year: int, month: str, id: int):
    data = updated_data_
    day = data[(data['eventyear'] == year) & (data['eventmonth'] == month)]
    if day.shape[0] < 600:
        return {"message": "Not Enough Records, Please Select Another Quarter", "status": 401}
    else:
        day = data[(data['eventyear'] == year) & (
            data['eventmonth'] == month)]
        # print('A preview at the selected month', day.head())
        # eventCount = len(day.event)
        # print(
        # f'The number of events made in the selected month is: {eventCount} events'
        # )
        # driverCount = day.DriverID.nunique()
        # print(
        # f"The number of drivers in this date range is: {driverCount}")
        eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            {"event": "count"}).add_prefix('Number of ')
        # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
        # print(
        # f'The average number of events made by the drivers is: {averageNoEvents}')

        fig = px.bar(eventsPerDriver.reset_index(),  x='Number of event',
                     y='DriverID', color='Number of event', barmode='group', orientation='h')
        fig.update_layout(
            yaxis_title="Number of Events",
            xaxis_title="driver ID",
            legend_title="DriverID",
            title="Bar Chart of Trips made per driver",
            template="plotly_white",
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/Number of Events by Driver.html')

        fig = px.histogram(eventsPerDriver.reset_index(),
                           x='Number of event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title="Histogram of Event Counts",
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/Histogram of Event Counts_.html')

        # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
        # DriverID = eventsPerDriver['Number of event'].idxmax()
        # print(
        # f'The driver with the most events is: driver {DriverID}, and the number of events made is: {maxEventsPerDriver}')
        # minEventsPerdriver = eventsPerDriver['Number of event'].min()
        # DriverID = eventsPerDriver['Number of event'].idxmin()
        # print(
        # f'The driver with the least events is: driver {DriverID}, and the number of events made is: {minEventsPerdriver}')
        # Event Type
        dfReasonHist = day.groupby(['event'])[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=True)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Count of Events",
            title='Bar Plot of Event Distribution',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/Event Distribution.html')

        # Handling Behavioral and Non-behavioral Events
        # non-behavioral events
        # non_behavioral_events = [event for event in day.event if event not in [
        #    'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # newCount = len(day[day['event'].isin(non_behavioral_events)])
        # print("Number of events before removing non-behavioral events is: {}.\nAfter removing non-behavioral events, we have: {} events.\nThis led to a reduction in the data size by: {:0.2f}%, leaving: {:0.2f}% of the entire data size.\nCurrent number of events is: {}".format(
        # len(day), newCount, ((len(day) - newCount)/len(day))*100, (100-(((len(day) - newCount)/len(day))*100)), newCount))
        # Specifying behavioral events
        behavioral_events = [event for event in day.event if event in [
            'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # Initializing the number of minimum events to 50 (intuitively)

        def prepData(day, minRecordsPerSubscriber=0):
            day.reset_index(inplace=True)
            # print(
            # f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
            # Remove NAs
            # df = day.dropna()
            # print(f"Removing NAs, we are left with: {len(df)} trips")
            # Filter out unwanted events
            df4 = day[day['event'].isin(behavioral_events)]
            # print(
            # f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
            # Filter out users with too few samples
            eventCountPerdriver = df4.groupby(
                'DriverID')['DriverID'].agg('count')
            driversWithManyRecords = eventCountPerdriver[eventCountPerdriver >
                                                         minRecordsPerSubscriber]
            driversWithManyRecords.keys()
            df5 = df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            # print(
            # f"Filtering users with too few samples,  we are left with: {len(df5)} trips")
            # print("*** Done. ***")
            return (df5)
        df6 = prepData(day)
        relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
            {"event": "count"}).sort_values(by='event', ascending=False)

        fig = px.histogram(relevantEventsPerSubscriber, x='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title='Histogram of Event Counts',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/Histogram of Event Counts.html')

        # Distribution of Events
        dfReasonHist = df6.groupby('event')[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=False)
        # print('Distribution of Events (in descending order)')
        # print(dfReasonHist)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Event",
            title='Distribution of Behavioral Events',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/Distribution of Behavioral Events.html')

        # Calculate distance traveled in each trip using Haversine
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        def total_distance(oneDriver):
            dist = haversine(oneDriver.longitude.shift(1), oneDriver.latitude.shift(1),
                             oneDriver.loc[1:, 'longitude'], oneDriver.loc[1:, 'latitude'])
            return np.sum(dist)
        # Calculate the overall distance made per driver

        def calculate_overall_distance_traveled(dfRaw):
            dfDistancePerdriver = day.groupby('DriverID').apply(
                total_distance).reset_index(name='Distance')
            return dfDistancePerdriver
        distancePerdriver = calculate_overall_distance_traveled(df6)
        # print('Distance Traveled per Driver (in descending order)')
        # print(distancePerdriver.sort_values(by='Distance', ascending=False))

        fig = px.bar(distancePerdriver.sort_values(by='Distance', ascending=True),
                     x='DriverID', y='Distance', color='Distance', )
        fig.update_layout(
            xaxis_title="Driver ID",
            yaxis_title="Distance Traveled",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            title='Distance Traveled per driver',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/Distance Traveled per driver.html')

        # Feature Engineering
        # Transform the events data frame to a features data frame (column for each type of relevant event)

        def create_feature_set(df6, distancePerdriver):
            dfEventAggByDriver = df6.groupby(['DriverID', 'event'])[['event']].agg(
                'count').add_prefix('Number of ').reset_index()
            # Pivot events into columns
            # Pivot the table by setting the drivers' name as the index column, while the respective events takes on a column each. Finally, fill missing observations with zeros(0s)
            dfEventMatrix = dfEventAggByDriver.pivot(index='DriverID', columns=[
                                                     'event'], values='Number of event').add_prefix('F_').fillna(0).reset_index()
            # Merge the created pivot table with the earlier created dataframe for distance traveled per driver.
            dfEventMatrix = dfEventMatrix.merge(
                distancePerdriver, how='inner', on='DriverID')
            dfEventMatrix.set_index('DriverID', inplace=True)
            # Set the columns to start with F_
            featureCols = [
                col for col in dfEventMatrix if col.startswith('F_')]
            # Divide each of the features by the traveled.
            dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                dfEventMatrix['Distance'], axis=0)
            dfFeatureSet = dfEventMatrix[featureCols]
            return dfFeatureSet
        features = create_feature_set(df6, distancePerdriver)
        # Motion based events
        try:
            if 'F_hardAcceleration' in features:
                features['F_hardAcceleration'] = features['F_hardAcceleration']
            else:
                features['F_hardAcceleration'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardBraking' in features:
                features['F_hardBraking'] = features['F_hardBraking']
            else:
                features['F_hardBraking'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardCornering' in features:
                features['F_hardCornering'] = features['F_hardCornering']
            else:
                features['F_hardCornering'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_overspeed' in features:
                features['F_overspeed'] = features['F_overspeed']
            else:
                features['F_overspeed'] = pd.Series(
                    [0 for x in range(len(features.index))])
            # features = [['F_hardAcceleration', 'F_hardBraking', 'F_hardCornering', 'F_overspeed']]
            features = features.fillna(0)
            # print(features)
        except Exception as e:
            print('Error')
            logger.warning(e)
            logger.error('Exception occurred', exc_info=True)

        features = features.rename(columns={'F_hardAcceleration': "Hard Acceleration",
                                            'F_hardBraking': "Hard Braking",
                                            'F_hardCornering': "Hard Cornering",
                                            'F_overspeed': 'Overspeed'}, inplace=False)
        # print(features.head())
        # Driver with the lowest harsh braking score
        # print(
        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmin()])
        # Driver with the highest harsh braking score
        # print(

        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmax()])

        fig = px.scatter_matrix(features.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()  # type: ignore
        fig.write_html(
            './HTML Charts/Year and Month and ID/scatter_matrix_features.html')

        # Detecting and Handling Outliers
        # Log Transformation
        features_log = np.log1p(features)
        features_log = features.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        features_log = features.fillna(0)
        # print('A preview of the data upon the application of Log Transformation')
        # print(features_log.head())

        fig = px.scatter_matrix(features_log.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/scatter_matrix_log_transformation.html')

        # Box-Cox Transformation
        # print('Applying Box-Cox transformation on the data')

        def transform_to_normal(x, min_max_transform=False):
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            valueGreaterThanZero = np.where(x <= 0, 0, x)
            positives = x[valueGreaterThanZero == 1]
            if (len(positives) > 0):
                xt[valueGreaterThanZero == 1], _ = st.boxcox(positives+1)
            if min_max_transform:
                xt = (xt - np.min(xt)) / (np.max(xt)-np.min(xt))
            return xt
        # features_1 = features_1.set_index('DriverId')
        transFeatures = features.apply(lambda x: (
            transform_to_normal(x, min_max_transform=True)))
        transFeatures = transFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeatures = transFeatures.fillna(0)
        # print('A preview of the data upon the application of Box-Cox Transformation')
        # print(transFeatures.head())

        fig = px.scatter_matrix(transFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/scatter_matrix_box_cox.html')

        # Standard Deviation Rule
        # print('Checking for and replacement of outliers')

        def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
            print(x.name)
            x = x.values
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            xt = transform_to_normal(x)
            xMean, xStd = np.mean(xt), np.std(xt)
            outliers = np.where(xt > xMean + stdFactor*xStd)[0]
            inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
            if len(outliers) > 0:
                # print("found outlier with factor: " +
                # str(stdFactor)+" : "+str(outliers))
                xinline = x[inliers]
                maxInRange = np.max(xinline)
                # print("replacing outliers {} with max={}".format(
                # outliers, maxInRange))
                vals = x.copy()
                vals[outliers] = maxInRange
                x = pd.Series(vals)
            else:
                pass
            if normalize:
                # Normalize to [0,1]
                x = (x - np.min(x)) / (np.max(x)-np.min(x))
            return x
        # features_1 = features_1.reset_index()
        cleanFeatures = features.apply(
            lambda x: (replace_outliers_with_limit(x)))
        cleanFeatures = cleanFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        cleanFeatures = cleanFeatures.fillna(0)
        # print("A preview of the cleaned data after handling outliers")
        # print(cleanFeatures.head())

        fig = px.scatter_matrix(cleanFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/scatter_matrix.html')

        # Correlation between events
        # print("Correlation between the events")
        corr = cleanFeatures.corr()
        corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
        corr = corr.fillna(0)
        # print(corr)

        # Plot the heatmap of the correlation matrix
       # print(f"""
        # Correlation Heatmap
        # """)
        fig = px.imshow(corr, color_continuous_scale='hot',
                        title='Correlation Heatmap', width=600, height=500, aspect='equal')
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Month and ID/correlation_heatmap.html')

        # Pre step: Normalize features
        # print("Data normalization")
        minPerFeature = cleanFeatures.min()
        maxPerFeature = cleanFeatures.max()
        # print("Min and Max values per column before normalization")
        # for col in range(0, len(cleanFeatures.columns)):
        # print(
        # f"{cleanFeatures.columns[col]} range:[{minPerFeature[col]},{maxPerFeature[col]}]")
        normalizedFeatures = (cleanFeatures-minPerFeature) / \
            (maxPerFeature-minPerFeature)
        normalizedFeatures = normalizedFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        normalizedFeatures = normalizedFeatures.fillna(0)
        # print("A preview of the normalized data")
        # print(normalizedFeatures.head())
        # Standardize features after box-cox as well.
        # print("Standardizing the features after Box-Cox Transformation")
        transFeaturesScaled = (
            transFeatures - transFeatures.mean())/transFeatures.std()
        transFeaturesScaled = transFeaturesScaled.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeaturesScaled = transFeaturesScaled.fillna(0)
        # print("A preview of the standardized data")
        # print(transFeaturesScaled.head())
        # print("Mean and STD before standardization")
        # for col in range(0, len(transFeatures.columns)):
        # print(
        # f"{transFeatures.columns[col]} range:[{transFeatures.mean()[col]},{transFeatures.std()[col]}]")
        # Anomaly Detection:
        # LOF - Local Outlier Filter
        # X = transFeaturesScaled.values
        # X = np.nan_to_num(X)
        # clf = LocalOutlierFactor(n_neighbors=5)
        # isOutlier = clf.fit_predict(X)
#
        # plt.title("Local Outlier Factor (LOF)", fontsize=20)
        # a = plt.scatter(X[isOutlier == 1, 0], X[isOutlier == 1, 1], c='white',
        #                edgecolor='k', s=40)
        # b = plt.scatter(X[isOutlier == -1, 0], X[isOutlier == -1, 1], c='red',
        #                edgecolor='k', s=40)
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # plt.xlabel(normalizedFeatures.columns[0], fontsize=15)
        # plt.ylabel(normalizedFeatures.columns[1], fontsize=15)
        # plt.xlim((-0.01, 1.01))
        # plt.ylim((-0.01, 1.01))
        # plt.legend([a, b],
        #           ["normal observations",
        #            "abnormal observations"],
        #           loc="upper right", prop={'size': 15}, frameon=True)
        # fig.show()
        # fig.write_html('./HTML Charts/Year and Month and ID/lof.html')

        # Multivariate analysis
        # Dimensionality reduction
        # PCA
        # pca = PCA(n_components=4)
        # principalComponents = pca.fit_transform(normalizedFeatures)
        # column_names = ['principal component {}'.format(
        #    i) for i in range(normalizedFeatures.shape[1])]
        # plt.bar(x=column_names, height=pca.explained_variance_ratio_)
        # plt.title("Percentage of explained variance")
        # fig.show()
        # print("Principal components explained variance ratio: {}.".format(
        # pca.explained_variance_ratio_))
        # principalDf = pd.DataFrame(
        #    data=principalComponents, columns=column_names)
        # df = normalizedFeatures
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # Show correlation matrix of the original features and the first principal component
        # pcAndOriginal = pd.concat(
        #    [principalDf.iloc[:, 0].reset_index(drop=True), normalizedFeatures], axis=1)
        # sns.set(style="ticks")
        # histplot = pcAndOriginal['principal component 0'].hist(figsize=(5, 5))
        # histplot.set_title("principal component 0 histogram")
        # sns.pairplot(pcAndOriginal, y_vars=['principal component 0'],
        #             x_vars=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'])
        # Extract statistics from the fitted distributions
        # normalizedFeatures.head()

        # Fit exponential distribution
        def fit_distribution_params(series):
            # print("Extracting distribution parameters for feature: " +
            #      series.name + " (" + str(len(series)) + ' values)')
            xPositive = series[series > 0]
            xPositive = xPositive.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            xPositive = xPositive.fillna(0)
            # xPositive = xPositive.replace
            probs = np.zeros(len(series))
            if (len(xPositive) > 0):
                params = st.expon.fit(xPositive)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # print('params = {}, {}, {}.'.format(arg, loc, scale))
                return arg, loc, scale

        def calculate_score_for_series(x, fittedParams, verbose=False):
            # print("Calculating scores for feature: " + x.name)
            xPositive = x[x > 0]
            probs = np.zeros(len(x))
            if (len(xPositive) > 0):
                arg = fittedParams[x.name]['arg']
                loc = fittedParams[x.name]['loc']
                scale = fittedParams[x.name]['scale']
                probs[x > 0] = st.expon.cdf(
                    xPositive, loc=loc, scale=scale, *arg)
                if verbose:
                    probs_df = pd.DataFrame(
                        {'Event value': x.values.tolist(), 'Event probability': probs}, index=True)
                    probs_df = probs_df.sort_values(by='Event value')
                    # print(probs_df)
            return probs
        # Store each fitted distribution parameters for later use
        fittedParams = {}
        for col in features.columns:
            arg, loc, scale = fit_distribution_params(features[col])
            fittedParams[col] = {}
            fittedParams[col]['arg'] = arg
            fittedParams[col]['loc'] = loc
            fittedParams[col]['scale'] = scale
        # print('Fitted parameters:')
        # print(json.dumps(fittedParams, indent=2))
        # Cumulative distribution/density function
        perFeatureScores = normalizedFeatures.apply(calculate_score_for_series, args=(
            fittedParams, False), axis=0).add_suffix("_CDF")
        # perFeatureScores.head()
        # DIST = st.expon

        # def create_pdf(dist, params, size=10000):
        #    # Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    start = dist.ppf(0.01, *arg, loc=loc,
        #                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        #    end = dist.ppf(0.99999, *arg, loc=loc,
        #                   scale=scale) if arg else dist.ppf(0.99999, loc=loc, scale=scale)
        #    x = np.linspace(start, end, size)
        #    y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #    pdf = pd.Series(y, x)
        #    return pdf

        # fit exponential distribution
        # fig, axs = plt.subplots(1, 4, figsize=(
        # 15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        # axs = axs.ravel()
        # i = 0
        # for col in normalizedFeatures:
        #    print(col)
        #    feature = normalizedFeatures[col]
        #    #only fit positive values to keep the distribution tighter
        #    x = feature.values[feature.values > 0]
        #    params = DIST.fit(x)
        #    #Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    #Plot
        #    pdfForPlot = create_pdf(DIST, params)
        #    pdfForPlot.plot()
        #    #Plot
        #    feature[feature.values > 0].plot(
        #    kind='hist', bins=30, )
        #    axs[i].set_ylabel('')
        #    axs[i].set_xlabel('')
        #    # Calculate SSE
        #    yhist, xhist = np.histogram(x, bins=60)
        #    xhist = (xhist + np.roll(xhist, -1))[:-1] / 2.0
        #    histPdf = DIST.pdf(xhist, loc=loc, scale=scale, *arg)
        #    sse = np.sum(np.power(yhist - histPdf, 2.0))
        #    print("sse:", sse)
        #    i += 1
        #    axs[1].set_xlabel('Events per km')
        #    axs[0].set_ylabel('Number of drivers')
        # Calculate driver score

        def calculate_joint_score(perFeatureScores):
            driverScores = perFeatureScores
            featureCols = [col for col in driverScores if col.startswith(
                'Hard') | col.startswith('Over')]
            driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                axis=1) / len(featureCols)))
            driverScores = driverScores.sort_values('score')
            driverScores['rank'] = len(
                driverScores['score']) - rankdata(driverScores['score']) + 1
            return driverScores
        driverScores = calculate_joint_score(perFeatureScores)
        driverScores = driverScores.reset_index()
        driverScores = driverScores.rename(columns={'DriverID': 'Driver ID', 'Hard Acceleration_CDF': 'Hard Acceleration', 'Hard Braking_CDF': 'Hard Braking',
                                           'Hard Cornering_CDF': 'Hard Cornering', 'Overspeed_CDF': 'Overspeed', 'score': 'Score', 'rank': 'Position'}, inplace=False)
        # print(driverScores.head())
        driverScores['Score'] = driverScores['Score']*100
        driverScores['Position'] = driverScores['Position']
        driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
        driverScores['Hard Braking'] = driverScores['Hard Braking']*100
        driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
        driverScores['Overspeed'] = driverScores['Overspeed']*100
        # print(driverScores)

        def condition(x):
            if x < 25:
                return "Perfectly Risky"
            elif x >= 25 and x < 50:
                return "Somewhat Risky"
            elif x >= 50 and x < 75:
                return 'Somewhat Safe'
            else:
                return 'Perfectly Safe'
        driverScores['Safety Class'] = driverScores['Score'].apply(
            condition)
        if id in driverScores['Driver ID'].values:
            driverScores1 = driverScores[driverScores['Driver ID'] == id]
            driverScores1 = driverScores1[[
                'Driver ID', 'Score', 'Position', 'Safety Class']]
            print("Score obtained per driver")
            print(driverScores1)
            result = driverScores1.to_json(orient="records")
            parsed = json.loads(result)
            json.dumps(parsed, indent=4)
            return {"message": "Success", "status": 200, "data": result}
        else:
            return {"message": "Driver ID not found", "status": 404}


@app.get('/getAnnualScorePerQuarter/{year}/{quarter}')
async def get_driver_score_per_year_per_quarter(year: int, quarter: int):
    day = updated_data_[(updated_data_['eventyear'] == year)
                        & (updated_data_['eventquarter'] == quarter)]

    if day.shape[0] < 600:
        return {"message": "Not Enough Records, Please Select Another Quarter", "status": 401}
    else:
        day = updated_data_[(updated_data_['eventyear'] == year) & (
            updated_data_['eventquarter'] == quarter)]
        print('A preview at the selected quarter', day.head())
        # eventCount = len(day.event)
        # print(
        # f'The number of events made in the selected month is: {eventCount} events'
        # )
        # driverCount = day.DriverID.nunique()
        # print(
        # f"The number of drivers in this date range is: {driverCount}")
        eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            {"event": "count"}).add_prefix('Number of ')
        # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
        # print(
        # f'The average number of events made by the drivers is: {averageNoEvents}')

        fig = px.bar(eventsPerDriver.reset_index(),  x='Number of event',
                     y='DriverID', color='Number of event', barmode='group', orientation='h')
        fig.update_layout(
            yaxis_title="Number of Events",
            xaxis_title="driver ID",
            legend_title="DriverID",
            title="Bar Chart of Trips made per Driver",
            template="plotly_white",
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Bar Chart of Trips made per Driver.html')

        fig = px.histogram(eventsPerDriver.reset_index(),
                           x='Number of event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title="Histogram of Event Counts",
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Histogram of Event.html')

        # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
        # DriverID = eventsPerDriver['Number of event'].idxmax()
        # print(
        # f'The driver with the most events is: driver {DriverID}, and the number of events made is: {maxEventsPerDriver}')
        # minEventsPerdriver = eventsPerDriver['Number of event'].min()
        # DriverID = eventsPerDriver['Number of event'].idxmin()
        # print(
        # f'The driver with the least events is: driver {DriverID}, and the number of events made is: {minEventsPerdriver}')
        # Event Type
        dfReasonHist = day.groupby(['event'])[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=True)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Count of Events",
            title='Bar Plot of Event Distribution',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Bar Plot of Event Distribution.html')

        # Handling Behavioral and Non-behavioral Events
        # non-behavioral events
        # non_behavioral_events = [event for event in day.event if event not in [
        #    'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # newCount = len(day[day['event'].isin(non_behavioral_events)])
        # print("Number of events before removing non-behavioral events is: {}.\nAfter removing non-behavioral events, we have: {} events.\nThis led to a reduction in the data size by: {:0.2f}%, leaving: {:0.2f}% of the entire data size.\nCurrent number of events is: {}".format(
        # len(day), newCount, ((len(day) - newCount)/len(day))*100, (100-(((len(day) - newCount)/len(day))*100)), newCount))
        # Specifying behavioral events
        behavioral_events = [event for event in day.event if event in [
            'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # Initializing the number of minimum events to 50 (intuitively)

        def prepData(day, minRecordsPerSubscriber=0):
            day.reset_index(inplace=True)
            # print(
            # f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
            # Remove NAs
            # df = day.dropna()
            # print(f"Removing NAs, we are left with: {len(df)} trips")
            # Filter out unwanted events
            df4 = day[day['event'].isin(behavioral_events)]
            # print(
            # f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
            # Filter out users with too few samples
            eventCountPerdriver = df4.groupby(
                'DriverID')['DriverID'].agg('count')
            driversWithManyRecords = eventCountPerdriver[eventCountPerdriver >
                                                         minRecordsPerSubscriber]
            driversWithManyRecords.keys()
            df5 = df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            # print(
            # f"Filtering users with too few samples,  we are left with: {len(df5)} trips")
            # print("*** Done. ***")
            return (df5)

        df6 = prepData(day)
        relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
            {"event": "count"}).sort_values(by='event', ascending=False)

        fig = px.histogram(relevantEventsPerSubscriber, x='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title='Histogram of Event Counts',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Histogram of Event Counts.html')

        # Distribution of Events
        dfReasonHist = df6.groupby('event')[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=False)
        # print('Distribution of Events (in descending order)')
        # print(dfReasonHist)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Event",
            title='Distribution of Behavioral Events',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Distribution of Behavioral Events.html')

        # Hard Acceleration Visual
        hard_acceleration = day[day['event'] == 'hardAcceleration']
        ha = hard_acceleration[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ha = ha.reset_index()
        ha = ha.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Acceleration Events per Driver')
        # print(ha)
        # print("")

        fig = px.bar(ha, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Acceleration per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Distribution of Hard Acceleration per Driver.html')

        # Hard Braking Visual
        hard_braking = day[day['event'] == 'hardBraking']
        hb = hard_braking[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hb = hb.reset_index()
        hb = hb.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Braking Events per Driver')
        # print(hb)
        # print("")

        fig = px.bar(hb, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Braking per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Distribution of Hard Braking per Driver.html')

        # Hard Cornering Visual
        hard_cornering = day[day['event'] == 'hardCornering']
        hc = hard_cornering[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        hc = hc.reset_index()
        hc = hc.sort_values(by='Number of event', ascending=False)
        # print('The number of Hard Cornering Events per Driver')
        # print(hc)
        # print("")

        fig = px.bar(hc, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Hard Cornering per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Distribution of Hard Cornering per Driver.html')

        # Overspeed Visual

        overspeed = day[day['event'] == 'overspeed']
        ovs = overspeed[['DriverID', 'event']].groupby('DriverID').agg('count').add_prefix(
            'Number of ')
        ovs = ovs.reset_index()
        ovs = ovs.sort_values(by='Number of event', ascending=False)
        # print('The number of Overspeeding Events per Driver')
        # print(ovs)
        # print("")

        fig = px.bar(ovs, x='DriverID', y='Number of event', color='Number of event',
                     title='Distribution of Overspeeding per Driver')
        fig.update_layout(
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Distribution of Overspeeding per Driver.html')

        # Calculate distance traveled in each trip using Haversine

        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        def total_distance(oneDriver):
            dist = haversine(oneDriver.longitude.shift(1), oneDriver.latitude.shift(1),
                             oneDriver.loc[1:, 'longitude'], oneDriver.loc[1:, 'latitude'])
            return np.sum(dist)

        # Calculate the overall distance made per driver

        def calculate_overall_distance_traveled(dfRaw):
            dfDistancePerdriver = day.groupby('DriverID').apply(
                total_distance).reset_index(name='Distance')
            return dfDistancePerdriver
        distancePerdriver = calculate_overall_distance_traveled(df6)
        # print('Distance Traveled per Driver (in descending order)')
        # print(distancePerdriver.sort_values(by='Distance', ascending=False))

        fig = px.bar(distancePerdriver.sort_values(by='Distance', ascending=True),
                     x='DriverID', y='Distance', color='Distance', )
        fig.update_layout(
            xaxis_title="Driver ID",
            yaxis_title="Distance Traveled",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            title='Distance Traveled per Driver',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Distance Traveled per Driver.html')
        # Feature Engineering
        # Transform the events data frame to a features data frame (column for each type of relevant event)

        def create_feature_set(df6, distancePerdriver):
            dfEventAggByDriver = df6.groupby(['DriverID', 'event'])[['event']].agg(
                'count').add_prefix('Number of ').reset_index()
            # Pivot events into columns
            # Pivot the table by setting the drivers' name as the index column, while the respective events takes on a column each. Finally, fill missing observations with zeros(0s)
            dfEventMatrix = dfEventAggByDriver.pivot(index='DriverID', columns=[
                                                     'event'], values='Number of event').add_prefix('F_').fillna(0).reset_index()
            # Merge the created pivot table with the earlier created dataframe for distance traveled per driver.
            dfEventMatrix = dfEventMatrix.merge(
                distancePerdriver, how='inner', on='DriverID')
            dfEventMatrix.set_index('DriverID', inplace=True)
            # Set the columns to start with F_
            featureCols = [
                col for col in dfEventMatrix if col.startswith('F_')]
            # Divide each of the features by the traveled.
            dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                dfEventMatrix['Distance'], axis=0)
            dfFeatureSet = dfEventMatrix[featureCols]
            return dfFeatureSet

        features = create_feature_set(df6, distancePerdriver)

        # Motion based events
        try:
            if 'F_hardAcceleration' in features:
                features['F_hardAcceleration'] = features['F_hardAcceleration']
            else:
                features['F_hardAcceleration'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardBraking' in features:
                features['F_hardBraking'] = features['F_hardBraking']
            else:
                features['F_hardBraking'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardCornering' in features:
                features['F_hardCornering'] = features['F_hardCornering']
            else:
                features['F_hardCornering'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_overspeed' in features:
                features['F_overspeed'] = features['F_overspeed']
            else:
                features['F_overspeed'] = pd.Series(
                    [0 for x in range(len(features.index))])
            # features = [['F_hardAcceleration', 'F_hardBraking', 'F_hardCornering', 'F_overspeed']]
            features = features.fillna(0)
            # print(features)
        except Exception as e:
            print('Error')
            logger.warning(e)
            logger.error('Exception occurred', exc_info=True)

        features = features.rename(columns={'F_hardAcceleration': "Hard Acceleration",
                                            'F_hardBraking': "Hard Braking",
                                            'F_hardCornering': "Hard Cornering",
                                            'F_overspeed': 'Overspeed'}, inplace=False)
        # print(features.head())

        # Driver with the lowest harsh braking score
        # print(
        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmin()])

        # Driver with the highest harsh braking score
        # print(

        #    f'The information about the driver with the least hard braking score is given below: ',

        #    features.loc[features['Hard Acceleration'].idxmax()])

        fig = px.scatter_matrix(features.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()  # type: ignore
        fig.write_html(
            './HTML Charts/Year and Quarter/scatter_matrix_features.html')

        # Detecting and Handling Outliers
        # Log Transformation
        features_log = np.log1p(features)
        features_log = features.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        features_log = features.fillna(0)
        # print('A preview of the data upon the application of Log Transformation')
        # print(features_log.head())

        fig = px.scatter_matrix(features_log.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/scatter_matrix_features_log.html')

        # Box-Cox Transformation
        # print('Applying Box-Cox transformation on the data')

        def transform_to_normal(x, min_max_transform=False):
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            valueGreaterThanZero = np.where(x <= 0, 0, x)
            positives = x[valueGreaterThanZero == 1]
            if (len(positives) > 0):
                xt[valueGreaterThanZero == 1], _ = st.boxcox(positives+1)
            if min_max_transform:
                xt = (xt - np.min(xt)) / (np.max(xt)-np.min(xt))
            return xt
        # features_1 = features_1.set_index('DriverId')
        transFeatures = features.apply(lambda x: (
            transform_to_normal(x, min_max_transform=True)))
        transFeatures = transFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeatures = transFeatures.fillna(0)
        # print('A preview of the data upon the application of Box-Cox Transformation')
        # print(transFeatures.head())

        fig = px.scatter_matrix(transFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/scatter_matrix_box_cox.html')

        # Standard Deviation Rule
        # print('Checking for and replacement of outliers')

        def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
            print(x.name)
            x = x.values
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            xt = transform_to_normal(x)
            xMean, xStd = np.mean(xt), np.std(xt)
            outliers = np.where(xt > xMean + stdFactor*xStd)[0]
            inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
            if len(outliers) > 0:
                # print("found outlier with factor: " +
                # str(stdFactor)+" : "+str(outliers))
                xinline = x[inliers]
                maxInRange = np.max(xinline)
                # print("replacing outliers {} with max={}".format(
                # outliers, maxInRange))
                vals = x.copy()
                vals[outliers] = maxInRange
                x = pd.Series(vals)
            else:
                pass
            if normalize:
                # Normalize to [0,1]
                x = (x - np.min(x)) / (np.max(x)-np.min(x))
            return x
        # features_1 = features_1.reset_index()
        cleanFeatures = features.apply(
            lambda x: (replace_outliers_with_limit(x)))
        cleanFeatures = cleanFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        cleanFeatures = cleanFeatures.fillna(0)
        # print("A preview of the cleaned data after handling outliers")
        # print(cleanFeatures.head())

        fig = px.scatter_matrix(cleanFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/scatter_matrix_cleaned_features.html')

        # Correlation between events
        # print("Correlation between the events")
        corr = cleanFeatures.corr()
        corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
        corr = corr.fillna(0)
        # print(corr)

        # Plot the heatmap of the correlation matrix
        # print(f"""
        #               ### Correlation Heatmap
        #                """)
        fig = px.imshow(corr, color_continuous_scale='hot',
                        title='Correlation Heatmap', width=600, height=500, aspect='equal')
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Correlation Heatmap.html')

        # Pre step: Normalize features
        # print("Data normalization")
        minPerFeature = cleanFeatures.min()
        maxPerFeature = cleanFeatures.max()
        # print("Min and Max values per column before normalization")
        # for col in range(0, len(cleanFeatures.columns)):
        #    print(
        #        f"{cleanFeatures.columns[col]} range:[{minPerFeature[col]},{maxPerFeature[col]}]")
        normalizedFeatures = (cleanFeatures-minPerFeature) / \
            (maxPerFeature-minPerFeature)
        normalizedFeatures = normalizedFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        normalizedFeatures = normalizedFeatures.fillna(0)
        # print("A preview of the normalized data")
        # print(normalizedFeatures.head())

        # Standardize features after box-cox as well.
        # print("Standardizing the features after Box-Cox Transformation")
        transFeaturesScaled = (
            transFeatures - transFeatures.mean())/transFeatures.std()
        transFeaturesScaled = transFeaturesScaled.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeaturesScaled = transFeaturesScaled.fillna(0)
        # print("A preview of the standardized data")
        # print(transFeaturesScaled.head())

        # print("Mean and STD before standardization")
        # for col in range(0, len(transFeatures.columns)):
        # print(
        # f"{transFeatures.columns[col]} range:[{transFeatures.mean()[col]},{transFeatures.std()[col]}]")

        # Anomaly Detection:
        # LOF - Local Outlier Filter
        # X = transFeaturesScaled.values
        # X = np.nan_to_num(X)
        # clf = LocalOutlierFactor(n_neighbors=5)
        # isOutlier = clf.fit_predict(X)
        # plt.title("Local Outlier Factor (LOF)", fontsize=20)
        # a = plt.scatter(X[isOutlier == 1, 0], X[isOutlier == 1, 1], c='white',
        #                edgecolor='k', s=40)
        # b = plt.scatter(X[isOutlier == -1, 0], X[isOutlier == -1, 1], c='red',
        #                edgecolor='k', s=40)
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # plt.xlabel(normalizedFeatures.columns[0], fontsize=15)
        # plt.ylabel(normalizedFeatures.columns[1], fontsize=15)
        # plt.xlim((-0.01, 1.01))
        # plt.ylim((-0.01, 1.01))
        # plt.legend([a, b],
        #           ["normal observations",
        #            "abnormal observations"],
        #           loc="upper right", prop={'size': 15}, frameon=True)
        # fig.show()
        # fig.write_html('./HTML Charts/Year and Quarter/lof.html')

        # Multivariate analysis
        # Dimensionality reduction
        # PCA
        # pca = PCA(n_components=4)
        # principalComponents = pca.fit_transform(normalizedFeatures)
        # column_names = ['principal component {}'.format(
        #    i) for i in range(normalizedFeatures.shape[1])]
        # plt.bar(x=column_names, height=pca.explained_variance_ratio_)
        # plt.title("Percentage of explained variance")
        # fig.show()
        # print("Principal components explained variance ratio: {}.".format(
        # pca.explained_variance_ratio_))
        # principalDf = pd.DataFrame(
        #    data=principalComponents, columns=column_names)
        # df = normalizedFeatures

        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # Show correlation matrix of the original features and the first principal component
        # pcAndOriginal = pd.concat(
        #    [principalDf.iloc[:, 0].reset_index(drop=True), normalizedFeatures], axis=1)
        # sns.set(style="ticks")
        # histplot = pcAndOriginal['principal component 0'].hist(figsize=(5, 5))
        # histplot.set_title("principal component 0 histogram")
        # sns.pairplot(pcAndOriginal, y_vars=['principal component 0'],
        #             x_vars=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'])
        # fig.write_html('./HTML Charts/Year and Quarter/pca.html')

        # Extract statistics from the fitted distributions
        # normalizedFeatures.head()

        # Fit exponential distribution
        def fit_distribution_params(series):
            # print("Extracting distribution parameters for feature: " +
            #      series.name + " (" + str(len(series)) + ' values)')
            xPositive = series[series > 0]
            xPositive = xPositive.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            xPositive = xPositive.fillna(0)
            # xPositive = xPositive.replace
            probs = np.zeros(len(series))
            if (len(xPositive) > 0):
                params = st.expon.fit(xPositive)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # print('params = {}, {}, {}.'.format(arg, loc, scale))
                return arg, loc, scale

        def calculate_score_for_series(x, fittedParams, verbose=False):
            # print("Calculating scores for feature: " + x.name)
            xPositive = x[x > 0]
            probs = np.zeros(len(x))
            if (len(xPositive) > 0):
                arg = fittedParams[x.name]['arg']
                loc = fittedParams[x.name]['loc']
                scale = fittedParams[x.name]['scale']
                probs[x > 0] = st.expon.cdf(
                    xPositive, loc=loc, scale=scale, *arg)
                if verbose:
                    probs_df = pd.DataFrame(
                        {'Event value': x.values.tolist(), 'Event probability': probs}, index=True)
                    probs_df = probs_df.sort_values(by='Event value')
                    # print(probs_df)
            return probs

        # Store each fitted distribution parameters for later use
        fittedParams = {}
        for col in features.columns:
            arg, loc, scale = fit_distribution_params(features[col])
            fittedParams[col] = {}
            fittedParams[col]['arg'] = arg
            fittedParams[col]['loc'] = loc
            fittedParams[col]['scale'] = scale
        # print('Fitted parameters:')
        # print(json.dumps(fittedParams, indent=2))

        # Cumulative distribution/density function
        perFeatureScores = normalizedFeatures.apply(calculate_score_for_series, args=(
            fittedParams, False), axis=0).add_suffix("_CDF")
        # perFeatureScores.head()
        # DIST = st.expon

        # def create_pdf(dist, params, size=10000):
        #    # Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    start = dist.ppf(0.01, *arg, loc=loc,
        #                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        #    end = dist.ppf(0.99999, *arg, loc=loc,
        #                   scale=scale) if arg else dist.ppf(0.99999, loc=loc, scale=scale)
        #    x = np.linspace(start, end, size)
        #    y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #    pdf = pd.Series(y, x)
        #    return pdf

        # fit exponential distribution
        # fig, axs = plt.subplots(1, 4, figsize=(
        # 15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        # axs = axs.ravel()
        # i = 0
        # for col in normalizedFeatures:
        #   print(col)
        #   feature = normalizedFeatures[col]
        #   #only fit positive values to keep the distribution tighter
        #   x = feature.values[feature.values > 0]
        #   params = DIST.fit(x)
        #   #Separate parts of parameters
        #   arg = params[:-2]
        #   loc = params[-2]
        #   scale = params[-1]
        #   #Plot
        #   pdfForPlot = create_pdf(DIST, params)
        #   pdfForPlot.plot()
        #   #Plot
        #   feature[feature.values > 0].plot(
        #   kind='hist', bins=30, )
        #   axs[i].set_ylabel('')
        #   axs[i].set_xlabel('')
        #   # Calculate SSE
        #   yhist, xhist = np.histogram(x, bins=60)
        #   xhist = (xhist + np.roll(xhist, -1))[:-1] / 2.0
        #   histPdf = DIST.pdf(xhist, loc=loc, scale=scale, *arg)
        #   sse = np.sum(np.power(yhist - histPdf, 2.0))
        #   print("sse:", sse)
        #   i += 1
        #   axs[1].set_xlabel('Events per km')
        #   axs[0].set_ylabel('Number of drivers')
        # fig.write_html('./HTML Charts/Year and Quarter/exponential_curve.html')

        # Calculate driver score
        def calculate_joint_score(perFeatureScores):
            driverScores = perFeatureScores
            featureCols = [col for col in driverScores if col.startswith(
                'Hard') | col.startswith('Over')]
            driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                axis=1) / len(featureCols)))
            driverScores = driverScores.sort_values('score')
            driverScores['rank'] = len(
                driverScores['score']) - rankdata(driverScores['score']) + 1
            return driverScores

        driverScores = calculate_joint_score(perFeatureScores)
        driverScores = driverScores.reset_index()
        driverScores = driverScores.rename(columns={'DriverID': 'Driver ID', 'Hard Acceleration_CDF': 'Hard Acceleration', 'Hard Braking_CDF': 'Hard Braking',
                                           'Hard Cornering_CDF': 'Hard Cornering', 'Overspeed_CDF': 'Overspeed', 'score': 'Score', 'rank': 'Position'}, inplace=False)
        # print(driverScores.head())
        driverScores['Score'] = driverScores['Score']*100
        driverScores['Position'] = driverScores['Position']
        driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
        driverScores['Hard Braking'] = driverScores['Hard Braking']*100
        driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
        driverScores['Overspeed'] = driverScores['Overspeed']*100
        # print(driverScores)

        def condition(x):
            if x < 25:
                return "Perfectly Risky"
            elif x >= 25 and x < 50:
                return "Somewhat Risky"
            elif x >= 50 and x < 75:
                return 'Somewhat Safe'
            else:
                return 'Perfectly Safe'
        driverScores['Safety Class'] = driverScores['Score'].apply(
            condition)

        driverScores1 = driverScores[['Driver ID',
                                     'Score', 'Position', 'Safety Class']]
        driverScores1 = driverScores1.sort_values(
            by='Score', ascending=False)

        result = driverScores1.to_json(orient="records")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        # print("Score obtained per driver")
        # print(driverScores1)

        fig = px.bar(driverScores, y='Score', x='Driver ID',
                     color='Safety Class',
                     hover_data=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed', 'Position'])
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            xaxis_title="Driver ID",
            yaxis_title='Score',
            title="Bar Chart of Driver Score",
            template='plotly_white'
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter/Bar Chart of Driver Scores.html')
        return {"message": "Success", "status": 200, "data": result}


@app.get('/getAnnualScorePerQuarterPerDriver/{year}/{quarter}/{id}')
async def get_driver_score_per_driver_per_year_per_quarter(year: int, quarter: int, id: int):
    data = updated_data_
    day = data[(data['eventyear'] == year) & (data['eventquarter'] == quarter)]
    if day.shape[0] < 600:
        return {"message": "Not Enough Records, Please Select Another Quarter", "status": 401}
    else:
        day = data[(data['eventyear'] == year) & (
            data['eventquarter'] == quarter)]
        # print('A preview at the selected month', day.head())
        # eventCount = len(day.event)
        # print(
        # f'The number of events made in the selected month is: {eventCount} events'
        # )
        # driverCount = day.DriverID.nunique()
        # print(
        # f"The number of drivers in this date range is: {driverCount}")
        eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            {"event": "count"}).add_prefix('Number of ')
        # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
        # print(
        # f'The average number of events made by the drivers is: {averageNoEvents}')

        fig = px.bar(eventsPerDriver.reset_index(),  x='Number of event',
                     y='DriverID', color='Number of event', barmode='group', orientation='h')
        fig.update_layout(
            yaxis_title="Number of Events",
            xaxis_title="driver ID",
            legend_title="DriverID",
            title="Bar Chart of Trips made per driver",
            template="plotly_white",
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/Number of Events by Driver.html')

        fig = px.histogram(eventsPerDriver.reset_index(),
                           x='Number of event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title="Histogram of Event Counts",
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/Histogram of Event Counts_.html')

        # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
        # DriverID = eventsPerDriver['Number of event'].idxmax()
        # print(
        # f'The driver with the most events is: driver {DriverID}, and the number of events made is: {maxEventsPerDriver}')
        # minEventsPerdriver = eventsPerDriver['Number of event'].min()
        # DriverID = eventsPerDriver['Number of event'].idxmin()
        # print(
        # f'The driver with the least events is: driver {DriverID}, and the number of events made is: {minEventsPerdriver}')
        # Event Type
        dfReasonHist = day.groupby(['event'])[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=True)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Count of Events",
            title='Bar Plot of Event Distribution',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/Event Distribution.html')

        # Handling Behavioral and Non-behavioral Events
        # non-behavioral events
        # non_behavioral_events = [event for event in day.event if event not in [
        #    'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # newCount = len(day[day['event'].isin(non_behavioral_events)])
        # print("Number of events before removing non-behavioral events is: {}.\nAfter removing non-behavioral events, we have: {} events.\nThis led to a reduction in the data size by: {:0.2f}%, leaving: {:0.2f}% of the entire data size.\nCurrent number of events is: {}".format(
        # len(day), newCount, ((len(day) - newCount)/len(day))*100, (100-(((len(day) - newCount)/len(day))*100)), newCount))
        # Specifying behavioral events
        behavioral_events = [event for event in day.event if event in [
            'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
        # Initializing the number of minimum events to 50 (intuitively)

        def prepData(day, minRecordsPerSubscriber=0):
            day.reset_index(inplace=True)
            # print(
            # f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
            # Remove NAs
            # df = day.dropna()
            # print(f"Removing NAs, we are left with: {len(df)} trips")
            # Filter out unwanted events
            df4 = day[day['event'].isin(behavioral_events)]
            # print(
            # f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
            # Filter out users with too few samples
            eventCountPerdriver = df4.groupby(
                'DriverID')['DriverID'].agg('count')
            driversWithManyRecords = eventCountPerdriver[eventCountPerdriver >
                                                         minRecordsPerSubscriber]
            driversWithManyRecords.keys()
            df5 = df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            # print(
            # f"Filtering users with too few samples,  we are left with: {len(df5)} trips")
            # print("*** Done. ***")
            return (df5)
        df6 = prepData(day)
        relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
            {"event": "count"}).sort_values(by='event', ascending=False)

        fig = px.histogram(relevantEventsPerSubscriber, x='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Number of drivers in Range",
            title='Histogram of Event Counts',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/Histogram of Event Counts.html')

        # Distribution of Events
        dfReasonHist = df6.groupby('event')[['event']].agg('count').add_prefix(
            'Number of ').reset_index().sort_values('Number of event', ascending=False)
        # print('Distribution of Events (in descending order)')
        # print(dfReasonHist)

        fig = px.bar(dfReasonHist, x='Number of event',
                     y='event', orientation='h', color='event')
        fig.update_layout(
            xaxis_title="Number of Events",
            yaxis_title="Event",
            title='Distribution of Behavioral Events',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/Distribution of Behavioral Events.html')

        # Calculate distance traveled in each trip using Haversine
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        def total_distance(oneDriver):
            dist = haversine(oneDriver.longitude.shift(1), oneDriver.latitude.shift(1),
                             oneDriver.loc[1:, 'longitude'], oneDriver.loc[1:, 'latitude'])
            return np.sum(dist)
        # Calculate the overall distance made per driver

        def calculate_overall_distance_traveled(dfRaw):
            dfDistancePerdriver = day.groupby('DriverID').apply(
                total_distance).reset_index(name='Distance')
            return dfDistancePerdriver
        distancePerdriver = calculate_overall_distance_traveled(df6)
        # print('Distance Traveled per Driver (in descending order)')
        # print(distancePerdriver.sort_values(by='Distance', ascending=False))

        fig = px.bar(distancePerdriver.sort_values(by='Distance', ascending=True),
                     x='DriverID', y='Distance', color='Distance', )
        fig.update_layout(
            xaxis_title="Driver ID",
            yaxis_title="Distance Traveled",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            title='Distance Traveled per driver',
            template="plotly_white"
        )
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/Distance Traveled per driver.html')

        # Feature Engineering
        # Transform the events data frame to a features data frame (column for each type of relevant event)

        def create_feature_set(df6, distancePerdriver):
            dfEventAggByDriver = df6.groupby(['DriverID', 'event'])[['event']].agg(
                'count').add_prefix('Number of ').reset_index()
            # Pivot events into columns
            # Pivot the table by setting the drivers' name as the index column, while the respective events takes on a column each. Finally, fill missing observations with zeros(0s)
            dfEventMatrix = dfEventAggByDriver.pivot(index='DriverID', columns=[
                                                     'event'], values='Number of event').add_prefix('F_').fillna(0).reset_index()
            # Merge the created pivot table with the earlier created dataframe for distance traveled per driver.
            dfEventMatrix = dfEventMatrix.merge(
                distancePerdriver, how='inner', on='DriverID')
            dfEventMatrix.set_index('DriverID', inplace=True)
            # Set the columns to start with F_
            featureCols = [
                col for col in dfEventMatrix if col.startswith('F_')]
            # Divide each of the features by the traveled.
            dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                dfEventMatrix['Distance'], axis=0)
            dfFeatureSet = dfEventMatrix[featureCols]
            return dfFeatureSet
        features = create_feature_set(df6, distancePerdriver)
        # Motion based events
        try:
            if 'F_hardAcceleration' in features:
                features['F_hardAcceleration'] = features['F_hardAcceleration']
            else:
                features['F_hardAcceleration'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardBraking' in features:
                features['F_hardBraking'] = features['F_hardBraking']
            else:
                features['F_hardBraking'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_hardCornering' in features:
                features['F_hardCornering'] = features['F_hardCornering']
            else:
                features['F_hardCornering'] = pd.Series(
                    [0 for x in range(len(features.index))])
            if 'F_overspeed' in features:
                features['F_overspeed'] = features['F_overspeed']
            else:
                features['F_overspeed'] = pd.Series(
                    [0 for x in range(len(features.index))])
            # features = [['F_hardAcceleration', 'F_hardBraking', 'F_hardCornering', 'F_overspeed']]
            features = features.fillna(0)
            # print(features)
        except Exception as e:
            print('Error')
            logger.warning(e)
            logger.error('Exception occurred', exc_info=True)

        features = features.rename(columns={'F_hardAcceleration': "Hard Acceleration",
                                            'F_hardBraking': "Hard Braking",
                                            'F_hardCornering': "Hard Cornering",
                                            'F_overspeed': 'Overspeed'}, inplace=False)
        # print(features.head())
        # Driver with the lowest harsh braking score
        # print(
        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmin()])
        # Driver with the highest harsh braking score
        # print(

        #    f'The information about the driver with the least hard braking score is given below: ',
        #    features.loc[features['Hard Acceleration'].idxmax()])

        fig = px.scatter_matrix(features.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()  # type: ignore
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/scatter_matrix_features')

        # Detecting and Handling Outliers
        # Log Transformation
        features_log = np.log1p(features)
        features_log = features.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        features_log = features.fillna(0)
        # print('A preview of the data upon the application of Log Transformation')
        # print(features_log.head())

        fig = px.scatter_matrix(features_log.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/scatter_matrix_log_transformation.html')

        # Box-Cox Transformation
        # print('Applying Box-Cox transformation on the data')

        def transform_to_normal(x, min_max_transform=False):
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            valueGreaterThanZero = np.where(x <= 0, 0, x)
            positives = x[valueGreaterThanZero == 1]
            if (len(positives) > 0):
                xt[valueGreaterThanZero == 1], _ = st.boxcox(positives+1)
            if min_max_transform:
                xt = (xt - np.min(xt)) / (np.max(xt)-np.min(xt))
            return xt
        # features_1 = features_1.set_index('DriverId')
        transFeatures = features.apply(lambda x: (
            transform_to_normal(x, min_max_transform=True)))
        transFeatures = transFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeatures = transFeatures.fillna(0)
        # print('A preview of the data upon the application of Box-Cox Transformation')
        # print(transFeatures.head())

        fig = px.scatter_matrix(transFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/scatter_matrix_box_cox.html')

        # Standard Deviation Rule
        # print('Checking for and replacement of outliers')

        def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
            print(x.name)
            x = x.values
            xt = np.zeros(len(x))
            if np.count_nonzero(x) == 0:
                # print("only zero valued values found")
                return x
            xt = transform_to_normal(x)
            xMean, xStd = np.mean(xt), np.std(xt)
            outliers = np.where(xt > xMean + stdFactor*xStd)[0]
            inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
            if len(outliers) > 0:
                # print("found outlier with factor: " +
                # str(stdFactor)+" : "+str(outliers))
                xinline = x[inliers]
                maxInRange = np.max(xinline)
                # print("replacing outliers {} with max={}".format(
                # outliers, maxInRange))
                vals = x.copy()
                vals[outliers] = maxInRange
                x = pd.Series(vals)
            else:
                pass
            if normalize:
                # Normalize to [0,1]
                x = (x - np.min(x)) / (np.max(x)-np.min(x))
            return x
        # features_1 = features_1.reset_index()
        cleanFeatures = features.apply(
            lambda x: (replace_outliers_with_limit(x)))
        cleanFeatures = cleanFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        cleanFeatures = cleanFeatures.fillna(0)
        # print("A preview of the cleaned data after handling outliers")
        # print(cleanFeatures.head())

        fig = px.scatter_matrix(cleanFeatures.reset_index(), dimensions=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'],
                                color='DriverID', width=800, height=800)
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/scatter_matrix.html')

        # Correlation between events
        # print("Correlation between the events")
        corr = cleanFeatures.corr()
        corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
        corr = corr.fillna(0)
        # print(corr)

        # Plot the heatmap of the correlation matrix
        # print(f"""
        # Correlation Heatmap
        # """)
        fig = px.imshow(corr, color_continuous_scale='hot',
                        title='Correlation Heatmap', width=600, height=500, aspect='equal')
        # fig.show()
        fig.write_html(
            './HTML Charts/Year and Quarter and ID/correlation_heatmap.html')

        # Pre step: Normalize features
        # print("Data normalization")
        minPerFeature = cleanFeatures.min()
        maxPerFeature = cleanFeatures.max()
        # print("Min and Max values per column before normalization")
        # for col in range(0, len(cleanFeatures.columns)):
        # print(
        # f"{cleanFeatures.columns[col]} range:[{minPerFeature[col]},{maxPerFeature[col]}]")
        normalizedFeatures = (cleanFeatures-minPerFeature) / \
            (maxPerFeature-minPerFeature)
        normalizedFeatures = normalizedFeatures.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        normalizedFeatures = normalizedFeatures.fillna(0)
        # print("A preview of the normalized data")
        # print(normalizedFeatures.head())
        # Standardize features after box-cox as well.
        # print("Standardizing the features after Box-Cox Transformation")
        transFeaturesScaled = (
            transFeatures - transFeatures.mean())/transFeatures.std()
        transFeaturesScaled = transFeaturesScaled.replace(
            [np.inf, -np.inf], np.nan, inplace=False)
        transFeaturesScaled = transFeaturesScaled.fillna(0)
        # print("A preview of the standardized data")
        # print(transFeaturesScaled.head())
        # print("Mean and STD before standardization")
        # for col in range(0, len(transFeatures.columns)):
        # print(
        # f"{transFeatures.columns[col]} range:[{transFeatures.mean()[col]},{transFeatures.std()[col]}]")
        # Anomaly Detection:
        # LOF - Local Outlier Filter
        # X = transFeaturesScaled.values
        # X = np.nan_to_num(X)
        # clf = LocalOutlierFactor(n_neighbors=5)
        # isOutlier = clf.fit_predict(X)
#
        # plt.title("Local Outlier Factor (LOF)", fontsize=20)
        # a = plt.scatter(X[isOutlier == 1, 0], X[isOutlier == 1, 1], c='white',
        #                edgecolor='k', s=40)
        # b = plt.scatter(X[isOutlier == -1, 0], X[isOutlier == -1, 1], c='red',
        #                edgecolor='k', s=40)
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # plt.xlabel(normalizedFeatures.columns[0], fontsize=15)
        # plt.ylabel(normalizedFeatures.columns[1], fontsize=15)
        # plt.xlim((-0.01, 1.01))
        # plt.ylim((-0.01, 1.01))
        # plt.legend([a, b],
        #           ["normal observations",
        #            "abnormal observations"],
        #           loc="upper right", prop={'size': 15}, frameon=True)
        # fig.show()
        # fig.write_html('./HTML Charts/Year and Month and ID/lof.html')

        # Multivariate analysis
        # Dimensionality reduction
        # PCA
        # pca = PCA(n_components=4)
        # principalComponents = pca.fit_transform(normalizedFeatures)
        # column_names = ['principal component {}'.format(
        #    i) for i in range(normalizedFeatures.shape[1])]
        # plt.bar(x=column_names, height=pca.explained_variance_ratio_)
        # plt.title("Percentage of explained variance")
        # fig.show()
        # print("Principal components explained variance ratio: {}.".format(
        # pca.explained_variance_ratio_))
        # principalDf = pd.DataFrame(
        #    data=principalComponents, columns=column_names)
        # df = normalizedFeatures
        # fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        # Show correlation matrix of the original features and the first principal component
        # pcAndOriginal = pd.concat(
        #    [principalDf.iloc[:, 0].reset_index(drop=True), normalizedFeatures], axis=1)
        # sns.set(style="ticks")
        # histplot = pcAndOriginal['principal component 0'].hist(figsize=(5, 5))
        # histplot.set_title("principal component 0 histogram")
        # sns.pairplot(pcAndOriginal, y_vars=['principal component 0'],
        #             x_vars=['Hard Acceleration', 'Hard Braking', 'Hard Cornering', 'Overspeed'])
        # Extract statistics from the fitted distributions
        # normalizedFeatures.head()

        # Fit exponential distribution
        def fit_distribution_params(series):
            # print("Extracting distribution parameters for feature: " +
            #      series.name + " (" + str(len(series)) + ' values)')
            xPositive = series[series > 0]
            xPositive = xPositive.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            xPositive = xPositive.fillna(0)
            # xPositive = xPositive.replace
            probs = np.zeros(len(series))
            if (len(xPositive) > 0):
                params = st.expon.fit(xPositive)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # print('params = {}, {}, {}.'.format(arg, loc, scale))
                return arg, loc, scale

        def calculate_score_for_series(x, fittedParams, verbose=False):
            # print("Calculating scores for feature: " + x.name)
            xPositive = x[x > 0]
            probs = np.zeros(len(x))
            if (len(xPositive) > 0):
                arg = fittedParams[x.name]['arg']
                loc = fittedParams[x.name]['loc']
                scale = fittedParams[x.name]['scale']
                probs[x > 0] = st.expon.cdf(
                    xPositive, loc=loc, scale=scale, *arg)
                if verbose:
                    probs_df = pd.DataFrame(
                        {'Event value': x.values.tolist(), 'Event probability': probs}, index=True)
                    probs_df = probs_df.sort_values(by='Event value')
                    # print(probs_df)
            return probs
        # Store each fitted distribution parameters for later use
        fittedParams = {}
        for col in features.columns:
            arg, loc, scale = fit_distribution_params(features[col])
            fittedParams[col] = {}
            fittedParams[col]['arg'] = arg
            fittedParams[col]['loc'] = loc
            fittedParams[col]['scale'] = scale
        # print('Fitted parameters:')
        # print(json.dumps(fittedParams, indent=2))
        # Cumulative distribution/density function
        perFeatureScores = normalizedFeatures.apply(calculate_score_for_series, args=(
            fittedParams, False), axis=0).add_suffix("_CDF")
        # perFeatureScores.head()
        # DIST = st.expon

        # def create_pdf(dist, params, size=10000):
        #    # Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    start = dist.ppf(0.01, *arg, loc=loc,
        #                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        #    end = dist.ppf(0.99999, *arg, loc=loc,
        #                   scale=scale) if arg else dist.ppf(0.99999, loc=loc, scale=scale)
        #    x = np.linspace(start, end, size)
        #    y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #    pdf = pd.Series(y, x)
        #    return pdf

        # fit exponential distribution
        # fig, axs = plt.subplots(1, 4, figsize=(
        # 15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        # axs = axs.ravel()
        # i = 0
        # for col in normalizedFeatures:
        #    print(col)
        #    feature = normalizedFeatures[col]
        #    #only fit positive values to keep the distribution tighter
        #    x = feature.values[feature.values > 0]
        #    params = DIST.fit(x)
        #    #Separate parts of parameters
        #    arg = params[:-2]
        #    loc = params[-2]
        #    scale = params[-1]
        #    #Plot
        #    pdfForPlot = create_pdf(DIST, params)
        #    pdfForPlot.plot()
        #    #Plot
        #    feature[feature.values > 0].plot(
        #    kind='hist', bins=30, )
        #    axs[i].set_ylabel('')
        #    axs[i].set_xlabel('')
        #    # Calculate SSE
        #    yhist, xhist = np.histogram(x, bins=60)
        #    xhist = (xhist + np.roll(xhist, -1))[:-1] / 2.0
        #    histPdf = DIST.pdf(xhist, loc=loc, scale=scale, *arg)
        #    sse = np.sum(np.power(yhist - histPdf, 2.0))
        #    print("sse:", sse)
        #    i += 1
        #    axs[1].set_xlabel('Events per km')
        #    axs[0].set_ylabel('Number of drivers')
        # Calculate driver score

        def calculate_joint_score(perFeatureScores):
            driverScores = perFeatureScores
            featureCols = [col for col in driverScores if col.startswith(
                'Hard') | col.startswith('Over')]
            driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                axis=1) / len(featureCols)))
            driverScores = driverScores.sort_values('score')
            driverScores['rank'] = len(
                driverScores['score']) - rankdata(driverScores['score']) + 1
            return driverScores
        driverScores = calculate_joint_score(perFeatureScores)
        driverScores = driverScores.reset_index()
        driverScores = driverScores.rename(columns={'DriverID': 'Driver ID', 'Hard Acceleration_CDF': 'Hard Acceleration', 'Hard Braking_CDF': 'Hard Braking',
                                           'Hard Cornering_CDF': 'Hard Cornering', 'Overspeed_CDF': 'Overspeed', 'score': 'Score', 'rank': 'Position'}, inplace=False)
        # print(driverScores.head())
        driverScores['Score'] = driverScores['Score']*100
        driverScores['Position'] = driverScores['Position']
        driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
        driverScores['Hard Braking'] = driverScores['Hard Braking']*100
        driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
        driverScores['Overspeed'] = driverScores['Overspeed']*100
        # print(driverScores)

        def condition(x):
            if x < 25:
                return "Perfectly Risky"
            elif x >= 25 and x < 50:
                return "Somewhat Risky"
            elif x >= 50 and x < 75:
                return 'Somewhat Safe'
            else:
                return 'Perfectly Safe'
        driverScores['Safety Class'] = driverScores['Score'].apply(
            condition)
        if id in driverScores['Driver ID'].values:
            driverScores1 = driverScores[driverScores['Driver ID'] == id]
            driverScores1 = driverScores1[[
                'Driver ID', 'Score', 'Position', 'Safety Class']]
            print("Score obtained per driver")
            print(driverScores1)
            result = driverScores1.to_json(orient="records")
            parsed = json.loads(result)
            json.dumps(parsed, indent=4)
            return {"message": "Success", "status": 200, "data": result}
        else:
            return {"message": "Driver ID not found", "status": 404}
