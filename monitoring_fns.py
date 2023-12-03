### Components for monitoring metrics, drift and retraining
import kfp
import argparse
from training_functions import *
from variables import *
from typing import NamedTuple
import pickle
from kfp.dsl.structures import yaml
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


# class FloatOutput(NamedTuple):
#     float_value: float

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-bigquery', 'pandas', 'pandas-gbq', 'google-cloud-aiplatform', 'pyarrow'],
    output_component_file='new_yamls/read_data_from_bq.yaml'
)
def read_bq_data(projectid: str, db:str, tablename: str, model_name:str,
                 train_data_filename: str,
                 output_df_path: Output[Dataset],
                 train_df_path: Output[Dataset]):
    from google.cloud import bigquery, aiplatform, storage
    from datetime import datetime, timedelta
    from pytz import timezone
    import pandas as pd
    import os

    tz = timezone('US/Pacific')
    end_date = str((datetime.now(tz) - timedelta(days=7)).date())
    client = storage.Client(projectid)
    models = aiplatform.Model.list(
    filter=f'display_name={model_name}',
    order_by="update_time",
    location='us-central1')
    latest_model = models[-1]
    model_path = latest_model.to_dict()['artifactUri']
    path_split = model_path.split('/')
    bucket_name = path_split[2]
    bucket = client.get_bucket(bucket_name)
    uri_folder = '/'.join(path_split[3:])
    train_df_blob = bucket.blob(f'{uri_folder}/{train_data_filename}')
    train_df_blob.download_to_filename(train_data_filename)

    train_df = pd.read_parquet(train_data_filename)

    # train_query = f'SELECT * FROM {projectid}.{db}.{tablename} where data_type = "Train"'
    query = f'SELECT * FROM {projectid}.{db}.{tablename} where report_datetime >= "{end_date}"'
    # train_df = pd.read_gbq(train_query, project_id=projectid, dialect= 'standard')
    train_df.to_csv(train_df_path.path, index=False)
    df = pd.read_gbq(query, project_id=projectid, dialect= 'standard')
    df.to_csv(output_df_path.path, index=False)

# @component(
#     base_image='python:3.9',
#     packages_to_install=['pandas', 'scikit-learn'],
#     output_component_file='new_yamls/get_accuracy.yaml'
# )
# def get_accuracy(df_path: Input[Dataset]) -> FloatOutput:
    
#     import pandas as pd
#     from sklearn.metrics import accuracy_score
#     import pickle

#     df = pd.read_csv(df_path.path)
#     accuracy = accuracy_score(df['Prediction'], df['incident_category'])
#     return FloatOutput(accuracy)
    
@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn'],
    output_component_file='new_yamls/get_accuracy.yaml'
)
def get_accuracy(df_path: Input[Dataset]) -> float:
    
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import pickle

    df = pd.read_csv(df_path.path)
    accuracy = accuracy_score(df['Prediction'], df['incident_category'])
    return accuracy
    
@component(
    base_image='python:3.9',
    packages_to_install= ['scipy', 'pandas', 'google-cloud-storage', 'google-cloud-aiplatform'],
    output_component_file='new_yamls/compare_distributions.yaml'
)
def get_and_compare_distributions(train_df_path: Input[Dataset], df_path: Input[Dataset], projectid: str, 
                                  accuracy:float, training_dist_dict_name: str, model_name: str, 
                                  eval_output: Output[Metrics]):

    import pandas as pd
    from scipy import stats
    from google.cloud import storage, aiplatform
    import pickle
    import os


    df = pd.read_csv(df_path.path)
    train_df = pd.read_csv(train_df_path.path)
    print ('df shape: ', df.shape)
    print ('train df shape ', train_df.shape)
    client = storage.Client(project=projectid)
    models = aiplatform.Model.list(filter=f'display_name={model_name}',
                                    order_by="update_time",
                                    location='us-central1')
    if len(models) > 0:
        latest_model = models[-1]
        model_path = latest_model.to_dict()['artifactUri']
        path_split = model_path.split('/')
        bucket_name = path_split[2]
        blob_name = os.path.join('/'.join(path_split[3:]), training_dist_dict_name)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(training_dist_dict_name)
        with open(training_dist_dict_name, 'rb') as file:
            train_dist_dict = pickle.load(file)
        cols_req = list(train_dist_dict.keys())
        for col in cols_req:
            df[col] = df.groupby('analysis_neighborhood', group_keys = False)[col].apply(lambda x: x.fillna(x.mean()))
            pvalue = stats.ks_2samp(train_df[col], df[col]).pvalue
            col_avg = df[col].mean()
            col_std = df[col].std()
            if pvalue < 0.05:
                print (f'Column {col} is experiencing Drift')
            eval_output.log_metric(f'{col}_pvalue', pvalue)
            eval_output.log_metric(f'{col}_average', col_avg)
            eval_output.log_metric(f'{col}_std_dev', col_std)
        eval_output.log_metric('Accuracy', accuracy)

@dsl.pipeline(
    name='SF Crime Monitor pipeline',
    description= 'Monitoring')
def sf_pipeline_monitor(projectid:str =projectid, dbname:str =datasetid, tablename:str =tablename,
                       train_df_name:str =train_df_name, model_name:str =model_name):
    read_data_monitor_task = read_bq_data(projectid= projectid, db= datasetid,
                                     tablename= tablename, train_data_filename= train_df_name,
                                     model_name=model_name)
    get_accuracy_task = get_accuracy(df_path= read_data_monitor_task.outputs['output_df_path'])
    compare_distributions_task = get_and_compare_distributions(train_df_path= read_data_monitor_task.outputs['train_df_path'],
                                    accuracy=get_accuracy_task.output,
                                    df_path= read_data_monitor_task.outputs['output_df_path'],
                                    projectid= projectid, training_dist_dict_name='data_drift_dict.pkl',
                                    model_name=model_name)
    
    acc = get_accuracy_task.output
    with dsl.If(acc < 1):
        sf_training_pipeline(projectid= projectid, datasetid= datasetid, tablename= tablename,
                        label_name= label_name, test_size=test_size, 
                        best_split_tries=best_split_tries, float_cols= float_cols, 
                        cat_cols= cat_cols, binary_cols= binary_cols, bucket_name= bucket_name,
                        accuracy_threshold= accuracy_threshold, model_name= model_name)

kfp.compiler.Compiler().compile(
    pipeline_func=sf_pipeline_monitor,
    package_path='new_yamls/monitoring.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow pipeline to run monitoring script and retrain if required.')
    parser.add_argument('type', help='Enter "yaml" if file needs to be saved or "run" if pipline needs to be run directly',
                       type=str)
    args = parser.parse_args()
    if args.type == 'yaml':
        if not os.path.exists('new_yamls'):
            os.mkdir('new_yamls')
        kfp.compiler.Compiler().compile(
            pipeline_func=sf_training_pipeline,
            package_path='new_yamls/monitoring_pipeline.yaml')
    elif args.type == 'run':
        client = Client(host= host_url)
        experiments = client.list_experiments().experiments
        experiment_names = list(map(lambda x: x.display_name, experiments))
        if training_experiment_name not in experiment_names:
            client.create_experiment(name=monitoring_experiment_name)
        client.create_run_from_pipeline_func(
        pipeline_func=sf_pipeline_monitor,
        experiment_name=monitoring_experiment_name)