import os
import argparse
from variables import *
from common_fns import *
from data_ingest_fns import preprocess_data, map_label, merge_with_nfl, add_dummy_cols
from kfp.dsl.structures import yaml
from kfp.client import Client
import kfp
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

@component(
    base_image='python:3.9',
    packages_to_install=['sodapy', 'pandas'],
    output_component_file='new_yamls/collect_batch_data.yaml'
)
def get_data(serving_data_path: Output[Dataset]):
    from datetime import datetime, timedelta
    from sodapy import Socrata
    import pandas as pd
    import pytz

    client = Socrata('data.sfgov.org', None)
    police_dataset_id = 'wg3w-h783'
    date_req = (datetime.now(pytz.timezone('US/Pacific')) - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
    results = client.get(police_dataset_id, query= f'select * where incident_date = "{date_req}"')
    if len(results) > 0:
        df = pd.DataFrame(results).drop('point', axis = 1).drop_duplicates('incident_id')
        # df['filed_online'] = 1
        df.to_csv(serving_data_path.path, index=False)

    else:
        print ('No Serving Data')


@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'google-cloud-storage', 'scikit-learn'],
    output_component_file = 'new_yamls/collect_batch_data.yaml'
)
def transform_cat_label_data(df_path: Input[Dataset], cat_encoder_path: str,
                             label_encoder_path: str, projectid: str,
                             bucket_name: str, blob_path_cat_enc: str,
                             blob_path_lab_enc: str, df_output_path: Output[Dataset],
                             label_output_path: Output[Artifact]):
    import pandas as pd
    import pickle
    from google.cloud import storage

    client = storage.Client(project=projectid)
    bucket = client.get_bucket(bucket_name)
    blob_cat = bucket.blob(blob_path_cat_enc)
    blob_lab_enc = bucket.blob(blob_path_lab_enc)
    blob_cat.download_to_filename(cat_encoder_path)
    blob_lab_enc.download_to_filename(label_encoder_path)

    with open(label_encoder_path, 'rb') as file:
        lab_enc = pickle.load(file)

    with open(cat_encoder_path, 'rb') as file:
        cat_enc = pickle.load(file)

    df = pd.read_csv(df_path.path)
    labels = df.pop('incident_category')
    for col in cat_enc.keys():
        df[col] = df[col].astype(str)
        new_values_index = df.loc[~df[col].isin(cat_enc[col].classes_)].index
        print (new_values_index)
        if len(new_values_index) > 0:
            print (f'New values found in {col} column: {df.loc[new_values_index, col].values.tolist()}')
        df.loc[new_values_index, col] = 'UNK'
        df[col] = cat_enc[col].transform(df[col])
    labels = lab_enc.transform(labels)
    df.to_csv(df_output_path.path)
    with open(label_output_path.path, 'wb') as file:
        pickle.dump(labels, file)

@component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas', 'google-cloud-storage',
                         'google-cloud-aiplatform', 'xgboost', 'pyarrow'],
    output_component_file='new_yamls/predict_serving.yaml'
)
def serving_predict(serving_data_path: Input[Dataset], labels_path: Input[Artifact],
                    model_name: str, req_columns: list, output_df_path: Output[Dataset],
                    projectid: str, cat_cols: list, float_cols: list,
                    binary_cols: list):
    from xgboost import XGBClassifier
    import pandas as pd
    from google.cloud import storage, aiplatform
    import pickle
    import os

    models = aiplatform.Model.list(
    filter=f'display_name={model_name}',
    order_by="update_time",
    location='us-central1')
    model = XGBClassifier()
    if len(models) > 0:
        serving_data = pd.read_csv(serving_data_path.path)[binary_cols+float_cols+cat_cols+req_columns]
        with open(labels_path.path, 'rb') as file:
            y_labels = pickle.load(file)
        latest_model = models[-1]
        model_path = latest_model.to_dict()['artifactUri']
        path_split = model_path.split('/')
        bucket_name = path_split[2]
        blob_name = os.path.join('/'.join(path_split[3:]), 'model.pkl')
        client = storage.Client(project=projectid)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename('model.pkl')
        model.load_model('model.pkl')
        y_pred = model.predict(serving_data.drop(req_columns, axis = 1).values)
        serving_data['Prediction'] = y_pred
        serving_data['incident_category'] = y_labels
        serving_data['data_type'] = 'non_ingest'
        serving_data.to_csv(output_df_path.path, index=False)

@dsl.pipeline(
    name='SF Crime predict pipeline',
    description= 'Reading data, using the same transformations and predict')
def sf_pipeline_predict(projectid:str =projectid, bucket_name:str =bucket_name, model_name:str =model_name,
                       float_cols:list =float_cols, cat_cols:list =cat_cols, binary_cols:list =binary_cols,
                       label_name:str =label_name, datasetid:str = datasetid, tableid:str = tablename, schema:dict =schema):
    get_serving_data_task = get_data()
    serving_preprocess_task = preprocess_data(df_path= get_serving_data_task.outputs['serving_data_path'], data='serving',
                                           to_keep = ['incident_id', 'police_district', 'incident_datetime',
                                      'latitude', 'longitude', 'filed_online', 'incident_day_of_week',
                                      'report_datetime', 'analysis_neighborhood', 'supervisor_district',
                                      'incident_category'])
    map_serving_label_task = map_label(df_path= serving_preprocess_task.outputs['df_output_path'])
    serving_merge_nfl_task = merge_with_nfl(sf_dataset = map_serving_label_task.outputs['mapped_df_path'])
    insert_dummy_cols_task = add_dummy_cols(df_path= serving_merge_nfl_task.outputs['merged_csv'])
    post_proc_serve_task = post_processing_after_split(input_df_path= insert_dummy_cols_task.outputs['output_df'])
    transform_cat_label_task = transform_cat_label_data(df_path=post_proc_serve_task.outputs['output_df_path'],
                                                        cat_encoder_path='cat_lab_encoder.pkl',
                                                        label_encoder_path='target_lab_encoder.pkl',
                                                        projectid=projectid,
                                                        bucket_name=bucket_name,
                                                        blob_path_cat_enc='cat_lab_encoder.pkl',
                                                        blob_path_lab_enc='target_lab_encoder.pkl')
    predict_serving_task = serving_predict(serving_data_path=transform_cat_label_task.outputs['df_output_path'],
            labels_path = transform_cat_label_task.outputs['label_output_path'], model_name=model_name, 
            req_columns = ['incident_id', 'report_datetime','incident_datetime', 'latitude', 'year', 'longitude', 'datetime', 
                           'incident_date', 'incident_time', 'gameday', 'gametime'],
            projectid=projectid, float_cols= float_cols, cat_cols= cat_cols, binary_cols= binary_cols)

    
    decode_serve_columns_task = decode_cols(df_path= predict_serving_task.outputs['output_df_path'], 
                                            projectid= projectid, bucket_name= bucket_name,
                target_col= label_name, cat_enc_path= 'cat_lab_encoder.pkl', label_enc_path= 'target_lab_encoder.pkl', 
                                            cols_to_decode= ['police_district', 'incident_day_of_week',
                                            'analysis_neighborhood', 'supervisor_district', 'month', 'hour',
                                            'latitude_buckets', 'longitude_buckets'])

    upload_serving_task = upload_to_bq(df_path = decode_serve_columns_task.outputs['output_df'], projectid= projectid,
                                datasetid= datasetid, tableid= tableid, location='US',
                                schema = schema)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow pipeline to run inference on new data.')
    parser.add_argument('type', help='Enter "yaml" if file needs to be saved or "run" if pipline needs to be run directly',
                       type=str)
    args = parser.parse_args()
    if args.type == 'yaml':
        if not os.path.exists('new_yamls'):
            os.mkdir('new_yamls')
        kfp.compiler.Compiler().compile(
            pipeline_func=sf_training_pipeline,
            package_path='new_yamls/serving_pipeline.yaml')
    elif args.type == 'run':
        client = Client(host= host_url)
        experiments = client.list_experiments().experiments
        experiment_names = list(map(lambda x: x.display_name, experiments))
        if serving_experiment_name not in experiment_names:
            client.create_experiment(name=serving_experiment_name)
        print ('HERE!!')
        client.create_run_from_pipeline_func(
        pipeline_func=sf_pipeline_predict,
        experiment_name=serving_experiment_name)