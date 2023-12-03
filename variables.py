host_url = <insert url here>
ingest_experiment_name = 'ingest_experiment'
training_experiment_name = 'train_experiment'
serving_experiment_name = 'serving_experiment'
monitoring_experiment_name = 'monitoring_experiment'
projectid = <insert project id>
datasetid = <insert datasetid>
tablename = <insert table name>
label_name = 'incident_category'
test_size=0.3
best_split_tries= 5
train_df_name= 'train_df.parquet.gzip'
float_cols= ['diff', 'till_game_time']
cat_cols= ['police_district', 'incident_day_of_week', 'analysis_neighborhood',
        'supervisor_district', 'month', 'hour', 'latitude_buckets', 'longitude_buckets']
binary_cols= ['is_between_sr_ss', 'is_weekend', 'is_game_day', 'filed_online']
bucket_name= <insert bucket name>
accuracy_threshold=0.3
model_name='xgboost_sf'
schema = {'incident_id': 'STRING',
                                          'report_datetime': 'STRING',
                                          'incident_datetime': 'STRING',
                                          'incident_date': 'STRING',
                                          'incident_time': 'STRING',
                                          'incident_category': 'STRING',
                                          'gametime': 'STRING',
                                          'datetime': 'STRING',
                                          'is_between_sr_ss': 'INTEGER',
                                            'is_weekend': 'INTEGER',
                                            'is_game_day': 'INTEGER',
                                            'filed_online': 'BOOL',
                                            'diff': 'FLOAT',
                                            'till_game_time': 'FLOAT',
                                            'police_district': 'STRING',
                                            'incident_day_of_week': 'STRING',
                                            'analysis_neighborhood': 'STRING',
                                            'supervisor_district': 'STRING',
                                            'month': 'STRING',
                                            'hour': 'STRING',
                                            'year': 'INTEGER',
                                            'latitude': 'FLOAT',
                                            'longitude': 'FLOAT',
                                            'gameday':'STRING',
                                            'latitude_buckets': 'STRING',
                                            'longitude_buckets': 'STRING',
                                            'Prediction': 'STRING',
                                            'data_type': 'STRING'}
