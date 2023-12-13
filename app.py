#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 18:07:08 2022
Last Update on Wed Dec 13 16:35 2023


@author: muhammad
"""


import warnings
import pymysql
import plotly.express as px
from streamlit_option_menu import option_menu
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import seaborn as sns
import scipy.stats as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import os
from dotenv import load_dotenv

load_dotenv()

plt.rcParams["figure.figsize"] = (20, 5)

# Suppress scientific notation
pd.options.display.float_format = '{:.2f}'.format
pd.options.plotting.backend = "plotly"

sns.set(style="ticks", font_scale=1.1)


warnings.filterwarnings("ignore")

paths = ['csv files', 'HTML Charts', 'HTML Charts/Year',
         'HTML Charts/Year and Month', 'HTML Charts/Year and Month and ID',
         'HTML Charts/Year and Quarter', 'HTML Charts/Year and Quarter and ID']


for path in paths:
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

with st.sidebar:
    choose = option_menu("Main Menu",
                         ["Navigation Guide",
                          "About",
                          "App",
                          "Contact"],
                         icons=['compass',
                                'house',
                                'app-indicator',
                                'person lines fill'],
                         menu_icon="list",
                         default_index=0,
                         styles={"container": {"padding": "5!important",
                                               "background-color": "#008000"},
                                 "icon": {"color": "orange",
                                          "font-size": "25px"},
                                 "nav-link": {"font-size": "16px",
                                              "text-align": "left",
                                              "margin": "0px",
                                              "--hover-color": "#ADD8E6"},
                                 "nav-link-selected": {
                                     "background-color": "#00008B"},
                                 })

if choose == "Navigation Guide":
    """ __Navigation Guide__"""
    st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
            </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Guide to using this app</p>',
                unsafe_allow_html=True)
    st.write(
        """
        ## Please make use of the sidebar to navigate through this app.
        """)

elif choose == "About":
    col1, col2 = st.columns([0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
            </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About the App</p>',
                    unsafe_allow_html=True)
    # with col2:               # To display brand log
    #    st.image(logo, width=130, caption="Twitter Logo")

    st.markdown("""

        <t> <b> WELCOME TO THE DRIVER SCORECARD SYSTEM DEMO APPLICATION </b> </t>

        <The> The Driver Scorecard System is an Artificial Intelligence (AI)
        powered system that scores drivers based on their driving behavior
        while in motion. The system intelligently score drivers by calculating the distance
        traveled by respective drivers (mapped by DriverID) within a specified
        date range, calculating the number of time the driver reported
        specific events such as hard acceleration, et al. </p>

        <The> The system goes ahead to perform advanced mathematical operations
        and scores the drivers. The system as well ranks each driver against other drivers. </p>

        """, unsafe_allow_html=True)

elif choose == 'App':

    # Add a description of the app
    st.markdown(""" <style> .font {
    font-size:25px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)
    st.markdown(
        '<p class="font">A Web App for Scoring Drivers based on their Driving Behavior</p>',
        unsafe_allow_html=True)  # use st.markdown() with CSS style to create a nice-formatted header/tex

    option = st.selectbox(
        'What task do you want to perform?',
        ('Fetch Updated Data', 'Obtain Driver Score by Event Month'),
        index=0
    )

    st.write('Your selected option is: ', option)

    if option == "Fetch Updated Data":

        get_data = st.button('Get data')

        if get_data:
            host = st.secrets.db_credentials.host
            user = st.secrets.db_credentials.user
            password = st.secrets.db_credentials.password
            db = st.secrets.db_credentials.database
            table1 = st.secrets.db_credentials.table1
            table2 = st.secrets.db_credentials.table2
            table3 = st.secrets.db_credentials.table3
            table4 = st.secrets.db_credentials.table4

            engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{db}')

            try:
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
                    engine
                )

                data = pd.DataFrame(
                    query, columns=['id', 'type', 'eventtime', 'deviceid',
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
                        st.warn(f"Error: {e}")
                        data_splitted.append('')

                data['event'] = data_splitted
                data['event'] = data['event'].apply(
                    lambda x: x.replace('"', ''))
                data['event'] = data['event'].apply(
                    lambda x: x.replace("}", ''))
                data['DriverID'] = data['deviceid']

                def part_of_day(x):
                    """
                    Returns the part of the day based on the given hour.

                    Args:
                        x (int): The hour of the day.

                    Returns:
                        str: The part of the day corresponding to the given hour.
                    """

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
                        ['DriverID', 'Company ID', 'Company Name', 'Email',
                         'positionid', 'timestamp', 'event', 'eventdate',
                         'eventtime', 'eventday', 'eventyear', 'eventmonth',
                         'eventweek', 'eventdayofweek', 'eventdayofmonth',
                         'eventdayofyear', 'eventquarter', 'eventhour',
                         'eventpartofday', 'latitude', 'longitude', 'altitude',
                         'speed', 'course', 'accuracy', ]]
                    
                    updated_data_ = updated_data

                test()

                updated_data_.drop(
                    ['Company ID', 'Company Name', 'Email', 'positionid',
                     'timestamp', 'course'],
                    axis=1).head().to_csv(
                        './csv files/data.csv', index=False,
                        header=True, encoding='utf-8')
                data.to_csv('./csv files/main_data.csv', index=False,
                            header=True, encoding='utf-8')

            except Exception as e:
                st.error(f"Error: {e}")

            # Create declarative base meta instance
            Base = declarative_base()
            # Create session local class for session maker
            SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

            @st.cache_data
            def convert_df(updated_data):
                """
                 # IMPORTANT: Cache the conversion to prevent computation on
                 # every rerun
                """
                return updated_data_.to_csv().encode('utf-8')
            
            csv = convert_df(updated_data_)

            st.write(f"""
                     ## A  preview of the fetched data
                     """)
            st.write(updated_data.drop(['Company ID', 'Company Name', 'Email', 'positionid', 'timestamp'], axis=1).head())

            # st.download_button(
            #   label="Download data as CSV",
            #  data=csv,
            # file_name='data.csv',
            # mime='text/csv',
            # )

    elif option == "Obtain Driver Score by Event Month":
        # Score Card System

        # st.subheader('Scorecard System')
        # option = st.selectbox(
        #    'Which option will you prefer?',
        #    ('Obtain Score by Event Date',
        #     'Obtain Rank by Event Date'),
        #    index=0)

        # Add a text input field to allow users to enter a search term
        updated_data = pd.read_csv('./csv files/data.csv')

        # st.subheader('Enter a date range to obtain the score')
        # today = date.today()
        # start_date = st.date_input(
        #    'Enter the start date of the events you want to score drivers on')
        # start_date = str(start_date)
        # end_date = st.date_input(
        #    'Enter the end date of the events you want to score drivers on')
        # end_date = str(end_date)

        # month = st.text_input('Enter the month of interest: ')
        # st.write(f"""
        #         ### Please select a month from June upwards
        #         """)
        option = st.selectbox(
            'Which month of year are you interested in?',
            ['January', 'February', 'March', 'April', 'May', 'June', 'July',
             'August', 'September', 'October', 'November', 'December'],
            index=0)
        st.write('You selected: ', option)
        if option == 'January':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'February':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'March':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'April':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'May':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'June':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'July':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'August':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'September':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'October':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'November':
            day = updated_data[updated_data['eventmonth'] == option]
        elif option == 'December':
            day = updated_data[updated_data['eventmonth'] == option]
        else:
            st.write('Please select a valid month option')
        day = updated_data[updated_data['eventmonth'] == option]
        if len(day['timestamp']) < 600:
            st.write('Please select another month')
        else:
            day = updated_data[updated_data['eventmonth'] == option]

            # st.write(
            # 'A preview at the selected month', day.drop(
            # ['Company ID', 'Company Name', 'Email', 'positionid', 'timestamp'],
            # axis=1).head())
            # eventCount = len(day.event)
            # st.write(
            #    f'''
            # The number of events made in the selected month is:
            # {eventCount} events'''
            # )
            # driverCount = day.DriverID.nunique()
            # st.write(
            #    f"The number of drivers in this date range is: {driverCount}")
            # eventsPerDriver = day.groupby('DriverID', as_index=True).agg(
            #    {"event": "count"}).add_prefix('Number of ')
            # averageNoEvents = np.mean(eventsPerDriver).values[0].round(2)
            # st.write(
            #    f'''
            # The average number of events made by the drivers is:
            # {averageNoEvents}''')

            # maxEventsPerDriver = eventsPerDriver['Number of event'].max()
            # DriverID = eventsPerDriver['Number of event'].idxmax()
            # st.write(
            #    f'The driver with the most events is: driver {DriverID}, and
            # the number of events made is: {maxEventsPerDriver}')
            # minEventsPerdriver = eventsPerDriver['Number of event'].min()
            # DriverID = eventsPerDriver['Number of event'].idxmin()
            # st.write(
            #    f'The driver with the least events is: driver {DriverID}, and
            # the number of events made is: {minEventsPerdriver}')
            # Event Type
            dfReasonHist = day.groupby(
                ['event'])[['event']].agg('count').add_prefix(
                'Number of ').reset_index().sort_values(
                    'Number of event', ascending=True)
            fig = px.bar(dfReasonHist, x='Number of event',
                         y='event', orientation='h', color='event')
            fig.update_layout(
                xaxis_title="Count of Events",
                title='Bar Plot of Event Distribution',
                template="plotly_white"
            )
            # st.plotly_chart(fig, use_container_width=True)
            # Handling Behavioral and Non-behavioral Events
            # non-behavioral events
            non_behavioral_events = [
                event for event in day.event if event not in [
                 'hardAcceleration', 'hardBraking',
                 'hardCornering', 'overspeed']]
            newCount = len(day[day['event'].isin(non_behavioral_events)])
            # st.write(
            # """Number of events before removing non-behavioral events is: {}.
            # After removing non-behavioral events, we have: {} events.
            # This led to a reduction in the data size by: {:0.2f}%, leaving:
            # {:0.2f}% of the entire data size.\nCurrent number of events is:
            # {}""".format(len(updated_data), newCount, (
            # (len(updated_data) - newCount)/len(updated_data))*100, (100-(((
            # len(updated_data) - newCount)/len(updated_data))*100)), newCount)
            # )
            # Specifying behavioral events
            behavioral_events = [event for event in day.event if event in [
                'hardAcceleration', 'hardBraking', 'hardCornering', 'overspeed']]
            # Initializing the number of minimum events to 50 (intuitively)

            def prepData(day, minRecordsPerSubscriber=0):
                day.reset_index(inplace=True)
                # st.write(
                #    f"*** Starting data prep, we have: {len(day)} trips in the dataset ***")
                # Remove NAs
                #df = day.dropna()
                # st.write(f"Removing NAs, we are left with: {len(df)} trips")
                # Filter out unwanted events
                df4 = day[day['event'].isin(behavioral_events)]
                # st.write(
                #    f"Keeping only events that are relevant for modeling, we are left with: {len(df4)} trips")
                # Filter out users with too few samples
                eventCountPerdriver = df4.groupby(
                    'DriverID')['DriverID'].agg('count')
                driversWithManyRecords = eventCountPerdriver[
                    eventCountPerdriver > minRecordsPerSubscriber]
                driversWithManyRecords.keys()
                return df4[df4['DriverID'].isin(driversWithManyRecords.keys())]
            df6 = prepData(day)
            relevantEventsPerSubscriber = df6.groupby('DriverID').agg(
                {"event": "count"}).sort_values(by='event', ascending=False)

            # Distribution of Events
            dfReasonHist = df6.groupby(
                'event')[['event']].agg(
                    'count').add_prefix('Number of ').reset_index(
                        ).sort_values('Number of event', ascending=True)
            fig = px.bar(
                dfReasonHist, x='Number of event',
                y='event', orientation='h', color='event')
            fig.update_layout(
                xaxis_title="Number of Events",
                yaxis_title="Event",
                title='Distribution of Behavioral Events',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Hard Acceleration Visual

            hard_acceleration = day[day['event'] == 'hardAcceleration']
            ha = hard_acceleration[
                ['DriverID', 'event']].groupby('DriverID').agg(
                    'count').add_prefix(
                'Number of ')
            ha = ha.reset_index()
            ha = ha.sort_values(by='Number of event', ascending=False)
            # st.write(ha.head())
            fig = px.bar(
                ha, x='DriverID', y='Number of event',
                color='Number of event',
                title='Distribution of Hard Acceleration per Driver')

            fig.update_layout(
                template='plotly_white'
            )
            # st.plotly_chart(fig, use_container_width=True)

            # Hard Braking Visual

            hard_braking = day[day['event'] == 'hardBraking']
            hb = hard_braking[
                ['DriverID', 'event']].groupby('DriverID').agg(
                    'count').add_prefix(
                'Number of ')
            hb = hb.reset_index()
            hb = hb.sort_values(by='Number of event', ascending=False)
            # st.write(ha.head())
            fig = px.bar(
                hb, x='DriverID', y='Number of event',
                color='Number of event',
                title='Distribution of Hard Braking per Driver')

            fig.update_layout(
                template='plotly_white'
            )
            # st.plotly_chart(fig, use_container_width=True)

            # Hard Cornering Visual

            hard_cornering = day[day['event'] == 'hardCornering']
            hc = hard_cornering[
                ['DriverID', 'event']].groupby('DriverID').agg(
                    'count').add_prefix('Number of ')
            hc = hc.reset_index()
            hc = hc.sort_values(by='Number of event', ascending=False)
            # st.write(ha.head())
            fig = px.bar(
                hc, x='DriverID',
                y='Number of event', color='Number of event',
                title='Distribution of Hard Cornering per Driver')

            fig.update_layout(
                template='plotly_white'
            )
            # st.plotly_chart(fig, use_container_width=True)

            # Overspeed Visual

            overspeed = day[day['event'] == 'overspeed']
            ovs = overspeed[
                ['DriverID', 'event']].groupby(
                    'DriverID').agg('count').add_prefix('Number of ')
            ovs = ovs.reset_index()
            ovs = ovs.sort_values(by='Number of event', ascending=False)
            # st.write(ha.head())
            fig = px.bar(ovs, x='DriverID', y='Number of event',
                         color='Number of event',
                         title='Distribution of Overspeeding per Driver')

            fig.update_layout(
                template='plotly_white'
            )
            # st.plotly_chart(fig, use_container_width=True)

            # Calculate distance traveled in each trip using Haversine

            def haversine(lon1, lat1, lon2, lat2):
                lon1, lat1, lon2, lat2 = map(
                    np.radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(
                    dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(
                        dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return 6367 * c

            def total_distance(oneDriver):
                dist = haversine(
                    oneDriver.longitude.shift(1),
                    oneDriver.latitude.shift(1),
                    oneDriver.loc[1:, 'longitude'],
                    oneDriver.loc[1:, 'latitude'])
                return np.sum(dist)
            # Calculate the overall distance made by each driver

            def calculate_overall_distance_traveled(dfRaw):
                return (
                    day.groupby('DriverID')
                    .apply(total_distance)
                    .reset_index(name='Distance')
                )
            distancePerdriver = calculate_overall_distance_traveled(df6)
            # st.write(distancePerdriver)
            fig = px.bar(
                distancePerdriver.sort_values(
                    by='Distance', ascending=True),
                x='DriverID',
                y='Distance',
                color='Distance', )
            fig.update_layout(
                xaxis_title="Driver ID",
                yaxis_title="Distance Traveled",
                barmode='stack',
                xaxis={'categoryorder': 'total descending'},
                title='Distance Traveled by each driver',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            # Feature Engineering
            # Transform the events data frame to a features data frame
            # (column for each type of relevant event)

            def create_feature_set(df6, distancePerdriver):
                dfEventAggBydriver = df6.groupby(
                    ['DriverID', 'event'])[['event']].agg(
                    'count').add_prefix('Number of ').reset_index()
                # Pivot events into columns
                # Pivot the table by setting the drivers' name as the index
                # column,while the respective events takes on a column each.
                # Finally,fill missing observations with zeros(0s)
                dfEventMatrix = dfEventAggBydriver.pivot(
                    index='DriverID',
                    columns=['event'],
                    values='Number of event').add_prefix(
                        'F_').fillna(0).reset_index()
                # Merge the created pivot table with the earlier created dataframe for distance traveled by each driver.
                dfEventMatrix = dfEventMatrix.merge(
                    distancePerdriver, how='inner', on='DriverID')
                dfEventMatrix.set_index('DriverID', inplace=True)
                # Set the columns to start with F_
                featureCols = [
                    col for col in dfEventMatrix if col.startswith('F_')]
                # Divide each of the features by the traveled.
                dfEventMatrix[featureCols] = dfEventMatrix[featureCols].div(
                    dfEventMatrix['Distance'], axis=0)
                return dfEventMatrix[featureCols]
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
                # features = [
                    # ['F_hardAcceleration',
                    # 'F_hardBraking',
                    # 'F_hardCornering',
                    # 'F_overspeed']
                    # ]
                features = features.fillna(0)
                # st.write(features)
            except Exception as e:
                st.error(f"{e}")
            features = features.rename(
                columns={
                    'F_hardAcceleration': "Hard Acceleration",
                    'F_hardBraking': "Hard Braking",
                    'F_hardCornering': "Hard Cornering",
                    'F_overspeed': 'Overspeed'},
                inplace=False)
            # st.write(features.head())

            # Detecting and Handling Outliers
            # Log Transformation
            features_log = np.log1p(features)
            features_log = features.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            features_log = features.fillna(0)

            # Box-Cox Transformation
            # st.write(f"""
            # ### Applying Box-Cox transformation on the data
            # """)

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
            # st.write(
            #    'A preview of the data upon the application of Box-Cox Transformation')
            # st.write(transFeatures.head())

            # Standard Deviation Rule
            # st.write(f'''
            #         ### Checking for and replacement of outliers''')

            def replace_outliers_with_limit(x, stdFactor=2.5, normalize=False):
                # st.write(x.name)
                x = x.values
                xt = np.zeros(len(x))
                if np.count_nonzero(x) == 0:
                    # st.write("only zero valued values found")
                    return x
                xt = transform_to_normal(x)
                xMean, xStd = np.mean(xt), np.std(xt)
                outliers = np.where(xt > xMean + stdFactor*xStd)[0]
                inliers = np.where(xt <= xMean + stdFactor*xStd)[0]
                if len(outliers) > 0:
                    # st.write("found outlier with factor: "
                    # +str(stdFactor)+" : "+str(outliers))
                    xinline = x[inliers]
                    maxInRange = np.max(xinline)
                    # st.write(f"""
                    # replacing outliers {outliers} with max={maxInRange}""")
                    vals = x.copy()
                    vals[outliers] = maxInRange
                    x = pd.Series(vals)
                else:
                    st.write("No outliers found")
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
            # st.write(f"""
            #         #### A preview of the cleaned data after handling outliers
            #         """)
            # st.write(cleanFeatures.head())

            # Correlation between events
            # st.write(f"""
            #         ### Correlation between the events
            #         """)
            # corr = cleanFeatures.corr()
            # corr = corr.replace([np.inf, -np.inf], np.nan, inplace=False)
            # corr = corr.fillna(0)
            # st.write(corr)

            # Draw the heatmap of the correlation matrix
            # st.write(f"""
            #        ### Correlation Heatmap
            #         """)
            # fig = px.imshow(corr, color_continuous_scale='hot',
            #                title='Correlation Heatmap',
            # width=600,
            # height=500,
            # aspect='equal')
            # st.plotly_chart(fig, use_container_width=True)
            # Pre step: Normalize features
            # st.write(f"""
            #         ### Data normalization
            #         """)
            minPerFeature = cleanFeatures.min()
            maxPerFeature = cleanFeatures.max()
            # st.write("Min and Max values per column before normalization")
            # for col in range(0, len(cleanFeatures.columns)):
            #    st.write(
            #        f"{cleanFeatures.columns[col]}
            # range:[{minPerFeature[col]},
            # {maxPerFeature[col]}]")
            normalizedFeatures = (cleanFeatures-minPerFeature) / \
                (maxPerFeature-minPerFeature)
            normalizedFeatures = normalizedFeatures.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            normalizedFeatures = normalizedFeatures.fillna(0)
            # st.write(f"""
            #         ### A preview of the normalized data
            #         """)
            # st.write(normalizedFeatures.head())
            #
            # Standardize features after box-cox as well.
            # st.write(f"""
            #         ### Standardizing the features after Box-Cox Transformation
            #         """)
            transFeaturesScaled = (
                transFeatures - transFeatures.mean())/transFeatures.std()
            transFeaturesScaled = transFeaturesScaled.replace(
                [np.inf, -np.inf], np.nan, inplace=False)
            transFeaturesScaled = transFeaturesScaled.fillna(0)
            # st.write(f"""
            # ### A preview of the standardized data
            # """)
            # st.write(transFeaturesScaled.head())
            # st.write("")
            # st.write("Mean and STD before standardization")
            # for col in range(0, len(transFeatures.columns)):
            #    st.write(
            #        f"{transFeatures.columns[col]}
            # range:[{transFeatures.mean()[col]},
            # {transFeatures.std()[col]}]")

            # Fit exponential distribution
            def fit_distribution_params(series):
                # st.write("Extracting distribution parameters for feature: " +
                #      series.name + " (" + str(len(series)) + ' values)')
                xPositive = series[series > 0]
                xPositive = xPositive.replace(
                    [np.inf, -np.inf], np.nan, inplace=False)
                xPositive = xPositive.fillna(0)
                # xPositive = xPositive.replace
                probs = np.zeros(len(series))
                if (len(xPositive) > 0):
                    params = sc.expon.fit(xPositive)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # st.write('params = {}, {}, {}.'.format(arg, loc, scale))
                    return arg, loc, scale

            def calculate_score_for_series(x, fittedParams, verbose=False):
                # st.write("Calculating scores for feature: " + x.name)
                xPositive = x[x > 0]
                probs = np.zeros(len(x))
                if (len(xPositive) > 0):
                    arg = fittedParams[x.name]['arg']
                    loc = fittedParams[x.name]['loc']
                    scale = fittedParams[x.name]['scale']
                    probs[x > 0] = sc.expon.cdf(
                        xPositive, loc=loc, scale=scale, *arg)
                    if verbose:
                        probs_df = pd.DataFrame(
                            {'Event value': x.values.tolist(),
                                'Event probability': probs},
                            index=True)
                        probs_df = probs_df.sort_values(by='Event value')
                        # st.write(probs_df)
                return probs
            # Store each fitted distribution parameters for later use
            fittedParams = {}
            for col in features.columns:
                arg, loc, scale = fit_distribution_params(features[col])
                fittedParams[col] = {}
                fittedParams[col]['arg'] = arg
                fittedParams[col]['loc'] = loc
                fittedParams[col]['scale'] = scale
            # st.write('Fitted parameters:')
            # st.write(json.dumps(fittedParams, indent=2))
            # Cumulative distribution/density function
            perFeatureScores = normalizedFeatures.apply(
                calculate_score_for_series, args=(
                    fittedParams, False),
                axis=0).add_suffix("_CDF")
            # perFeatureScores.head()
            DIST = sc.expon

            def create_pdf(dist, params, size=10000):
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                start = dist.ppf(
                    0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(
                        0.01, loc=loc, scale=scale)
                end = dist.ppf(
                    0.99999, *arg, loc=loc, scale=scale) if arg else dist.ppf(
                        0.99999, loc=loc, scale=scale)
                x = np.linspace(start, end, size)
                y = dist.pdf(x, loc=loc, scale=scale, *arg)
                pdf = pd.Series(y, x)
                return pdf

            # Calculate driver score
            def calculate_joint_score(perFeatureScores):
                driverScores = perFeatureScores
                featureCols = [col for col in driverScores if col.startswith(
                    'Hard') | col.startswith('Over')]
                driverScores['score'] = 1 - ((driverScores[featureCols].sum(
                    axis=1) / len(featureCols)))
                driverScores = driverScores.sort_values('score')
                driverScores['rank'] = (len(
                    driverScores['score'])
                 - rankdata(driverScores['score']) + 1)
                return driverScores

            driverScores = calculate_joint_score(perFeatureScores)
            driverScores = driverScores.reset_index()
            driverScores = driverScores.rename(
                columns={
                    'DriverID': 'Driver ID',
                    'Hard Acceleration_CDF': 'Hard Acceleration',
                    'Hard Braking_CDF': 'Hard Braking',
                    'Hard Cornering_CDF': 'Hard Cornering',
                    'Overspeed_CDF': 'Overspeed',
                    'score': 'Driver Score',
                    'rank': 'Driver Rank'},
                inplace=False)
            # st.write(driverScores.head())
            driverScores['Driver Score'] = driverScores['Driver Score']*100
            driverScores['Driver Rank'] = driverScores['Driver Rank']
            driverScores['Hard Acceleration'] = driverScores['Hard Acceleration']*100
            driverScores['Hard Braking'] = driverScores['Hard Braking']*100
            driverScores['Hard Cornering'] = driverScores['Hard Cornering']*100
            driverScores['Overspeed'] = driverScores['Overspeed']*100
            # st.write(driverScores)

            def condition(x):
                if x < 25:
                    return "Perfectly Risky"
                elif x < 50:
                    return "Somewhat Risky"
                elif x < 75:
                    return 'Somewhat Safe'
                else:
                    return 'Perfectly Safe'
            driverScores['Safety Class'] = driverScores['Driver Score'].apply(
                condition)

            driverScores1 = driverScores[[
                'Driver ID', 'Driver Score', 'Driver Rank', 'Safety Class']]
            driverScores1 = driverScores1.sort_values(
                by='Driver Score', ascending=False)

            st.write("""
            ### Score obtained by each driver
            """)
            st.write(driverScores1)

            fig = px.bar(
                driverScores,
                y='Driver Score',
                x='Driver ID',
                color='Safety Class',
                hover_data=[
                    'Hard Acceleration', 'Hard Braking',
                    'Hard Cornering', 'Overspeed', 'Driver Rank']
                )
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            fig.update_layout(
                barmode='stack',
                xaxis={'categoryorder': 'total descending'},
                xaxis_title="Driver ID",
                yaxis_title='Driver Score',
                title="Bar Chat of Driver Score",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)