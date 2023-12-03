import argparse
from variables import *
from common_fns import *
from datetime import datetime, timedelta
import pytz
import kfp
from kfp.dsl.structures import yaml
from kfp.client import Client
import os
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics
)

pwd = os.getcwd()
if not os.path.exists(os.path.join(pwd, 'new_yamls')):
    os.mkdir(os.path.join(pwd, 'new_yamls'))


@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy'],
    output_component_file='new_yamls/preprocess_data.yaml'
)
def preprocess_data(df_path: Input[Dataset], to_keep: list, data:str,
                    df_output_path: Output[Dataset]):
    import pandas as pd
    from datetime import datetime, timedelta
    import pytz
    import numpy as np

    def get_year_month(input):
        return input.year, input.month, input.hour

    def get_date_time(input):
        return input.date(), input.time()
    df = pd.read_csv(df_path.path)
    df.drop_duplicates('incident_id', inplace = True)
    df = df[df['incident_category'] != 'Case Closure']
    if 'filed_online' not in df.columns:
        df['filed_online'] = False
    df['filed_online'].fillna(False, inplace = True)
    df = df[to_keep]
    df = df.dropna().reset_index(drop=True)
    df = df[df['police_district'] != 'Out of SF']
    if data == 'serving':
        time_format = None
    else:
        time_format = '%Y/%m/%d %I:%M:%S %p'
    for col in ['incident_datetime', 'report_datetime']:
        df[col] = pd.to_datetime(df[col], format = time_format)
    today = datetime.now(tz= pytz.timezone('US/Pacific'))
    max_date_training = (today - timedelta(days=3)).date()
    df['diff'] = (df['report_datetime'] - df['incident_datetime']).apply(lambda x: x.total_seconds()/3600)
    df = df[df['diff'] >= 0]
    df['diff'] = df['diff'].apply(lambda x: np.log(min(x, 365*24) + 1))
    df['incident_date'], df['incident_time'] = zip(*df['incident_datetime'].apply(lambda x: get_date_time(x)))
    if data == 'training':
        df = df[df['incident_date'] <= max_date_training]
    df['year'], df['month'], df['hour'] = zip(*df['incident_datetime'].apply(lambda x: get_year_month(x)))
    print (df['year'].unique())
    df['is_weekend'] = df['incident_day_of_week'].apply(lambda x: int(x in ['Saturday', 'Sunday']))
    # df = df[df['year'].isin([2018,2023])]
    df.to_csv(df_output_path.path, index=False)


@component(
    base_image='python:3.9',
    packages_to_install=['pandas'],
    output_component_file='new_yamls/mapping_labels.yaml'
)
def map_label(df_path: Input[Dataset], mapped_df_path: Output[Dataset]):
    import pandas as pd

    df = pd.read_csv(df_path.path)
    label_mapping = {'Human Trafficking (B), Involuntary Servitude': 'Human Trafficking',
                     'Human Trafficking, Commercial Sex Acts':'Human Trafficking',
                     'Human Trafficking (A), Commercial Sex Acts':'Human Trafficking',
                     'Weapons Offense':'Weapons Offence',
                     'Motor Vehicle Theft?':'Motor Vehicle Theft',
                     'Suspicious': 'Suspicious Occ',
                     'Other Offenses': 'Other',
                     'Other Miscellaneous': 'Other',
                     'Drug Offense': 'Drug Violation'}
    df['incident_category'] = df['incident_category'].replace(label_mapping)
    all_crimes = list(df['incident_category'].unique())
    crimes_dict = {'Other': 'other'}
    violent_crimes = ['Assault', 'Rape', 'Arson', 'Suicide', 'Weapons Offence', 'Homicide',
                        'Sex Offense', 'Human Trafficking']
    non_violent_crimes = ["Lost Property","Non-Criminal","Warrant","Suspicious Occ","Missing Person",
                          "Disorderly Conduct","Larceny Theft","Fire Report","Fraud","Malicious Mischief","Robbery",
                          "Burglary","Vandalism","Traffic Collision","Drug Violation","Motor Vehicle Theft",
                          "Traffic Violation Arrest","Recovered Vehicle","Miscellaneous Investigation","Weapons Carrying Etc",
                          "Forgery And Counterfeiting","Embezzlement","Stolen Property","Vehicle Misplaced",
                          "Offences Against The Family And Children","Prostitution","Vehicle Impounded",
                          "Courtesy Report","Liquor Laws","Gambling","Civil Sidewalks"]

    for crime in all_crimes:
        if crime in violent_crimes:
            crimes_dict[crime] = 'violent'
        elif crime in non_violent_crimes:
            crimes_dict[crime] = 'non violent'
        else:
            crimes_dict[crime] = 'other'

    df['incident_category'] = df['incident_category'].map(crimes_dict)
    df.to_csv(mapped_df_path.path, index=False)

@component(
    base_image='python:3.9',
    packages_to_install=['nfl-data-py', 'pandas', 'numpy', 'astral'],
    output_component_file='new_yamls/merging_crime_nfl_data.yaml')
def merge_with_nfl(sf_dataset: Input[Dataset], merged_csv: Output[Dataset]):
    import nfl_data_py as nfl
    import pandas as pd
    import numpy as np
    from astral import LocationInfo
    from astral.sun import sun
    from astral.location import Location
    from datetime import date

    def get_diff_in_time(input):
        game_time = input['datetime']
        incident_time = input['incident_datetime']
        if input['is_game_day'] == 0:
            return 0
        else:
            if game_time > incident_time:
                return (game_time-incident_time).total_seconds()//60
            return (-1 * ((incident_time-game_time).total_seconds()//60))

    def game_date_time(datetime):
        date = pd.to_datetime(datetime['gameday'], format= '%Y-%m-%d').date()
        time = pd.to_datetime(datetime['gametime'], format= '%H:%M').time()
        return time, date

    def get_sunrise_sunset(input, city, sf):
        incident_dt = input['incident_datetime']
        incident_time = input['incident_time']
        s = sun(city.observer, date=incident_dt, tzinfo=sf.timezone)
        sr = s['sunrise'].replace(tzinfo=None).time()
        ss = s['sunset'].replace(tzinfo=None).time()
        if incident_time <= sr or incident_time >= ss:
            return 1
        return 0

    lat, lon = [37.7, -122.452]
    city = LocationInfo("San Francisco", "United States", "US/Pacific", lat, lon)
    sf = Location(city)
    df = pd.read_csv(sf_dataset.path)
    all_years = df['year'].unique().tolist()

    nfl_data = nfl.import_schedules(all_years).query('home_team == "SF"')[['gameday', 'gametime']]
    nfl_data['datetime'] = pd.to_datetime(nfl_data['gameday'].astype(str) + ' ' + nfl_data['gametime'].astype(str), format='%Y-%m-%d %H:%M')
    nfl_data['gameday'] = pd.to_datetime(nfl_data['gameday']).apply(lambda x: x.date())
    print ('data type of date in nfl data: ', type(nfl_data['gameday'].tolist()[0]))
    print ('nfl shape: ', nfl_data.shape)
    print ('data type of date in orig data: ', type(df['incident_date'].tolist()[0]))
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    df['incident_date'] = pd.to_datetime(df['incident_date']).apply(lambda x: x.date())
    df['incident_time'] = pd.to_datetime(df['incident_time']).apply(lambda x: x.time())
    df = df.merge(nfl_data, how = 'left', right_on='gameday', left_on='incident_date')
    print ('number of game dates that are not null: ', df.query('gameday==gameday').shape)
    df.loc[pd.notnull(df['gameday']), 'is_game_day'] = 1
    df.loc[pd.isnull(df['gameday']), 'is_game_day'] = 0
    print (df['is_game_day'].value_counts())
    df['till_game_time'] = df.apply(lambda x: get_diff_in_time(x), axis = 1)
    df['is_between_sr_ss'] = df.apply(lambda x: get_sunrise_sunset(x, city, sf), axis = 1)
    df.to_csv(merged_csv.path, index=False)

@component(
    base_image='python:3.9',
    packages_to_install=['pandas'],
    output_component_file='new_yamls/add_dummy_cols.yaml'
)
def add_dummy_cols(df_path: Input[Dataset], output_df: Output[Dataset]):
    import pandas as pd

    df = pd.read_csv(df_path.path)
    dummy_cols = ['latitude_buckets', 'longitude_buckets', 'Prediction']
    for col in dummy_cols:
        df[col] = ''

    df.to_csv(output_df.path, index=False)
    
@dsl.pipeline(
    name='SF Data Pipeline',
    description= 'The entire pipeline from reading sf and NFL data to pre-processing it and uploading to BQ')
def sf_ingest_pipeline(url: str, projectid:str = projectid, datasetid:str =datasetid, tablename:str =tablename):

    get_data_task = read_training_data(url = url, method='ingesting', projectid=projectid, datasetid=datasetid,
                                       tablename=tablename)
    preprocess_data_task = preprocess_data(df_path= get_data_task.outputs['output_csv'], data = 'training',
                                           to_keep = ['incident_id', 'police_district', 'incident_datetime',
                                      'latitude', 'longitude', 'filed_online', 'incident_day_of_week',
                                      'report_datetime', 'analysis_neighborhood', 'supervisor_district',
                                      'incident_category', 'data_type'])
    map_label_task = map_label(df_path= preprocess_data_task.outputs['df_output_path'])
    merge_nfl_task = merge_with_nfl(sf_dataset = map_label_task.outputs['mapped_df_path'])
    insert_dummy_cols_task = add_dummy_cols(df_path= merge_nfl_task.outputs['merged_csv'])
    upload_task = upload_to_bq(df_path = insert_dummy_cols_task.outputs['output_df'], projectid= projectid,
                                datasetid= datasetid, tableid= tablename, location='US',
                                schema = schema)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow pipeline to run ingestion script for SF Crime data.')
    parser.add_argument('type', help='Enter "yaml" if file needs to be saved or "run" if pipline needs to be run directly',
                       type= str)
    args = parser.parse_args()
    
    if args.type == 'yaml':
        kfp.compiler.Compiler().compile(
            pipeline_func=sf_ingest_pipeline,
            package_path='new_yamls/ingestion_pipeline.yaml')
    elif args.type == 'run':
        client = Client(host= host_url)
        today = datetime.now(tz= pytz.timezone('US/Pacific'))
        max_date_training = (today - timedelta(days=3)).date().strftime('%Y%m%d')
        url = f'https://data.sfgov.org/api/views/wg3w-h783/rows.csv?date={max_date_training}&accessType=DOWNLOAD'
        experiments = client.list_experiments().experiments
        experiment_names = list(map(lambda x: x.display_name, experiments))
        if ingest_experiment_name not in experiment_names:
            client.create_experiment(name=ingest_experiment_name)
        client.create_run_from_pipeline_func(
        pipeline_func=sf_ingest_pipeline,
        experiment_name=ingest_experiment_name,
        arguments={
            'url': url,
        })
