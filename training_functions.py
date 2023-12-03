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


@component(
    base_image='python:3.9',
    packages_to_install=['numpy', 'scikit-learn', 'pandas', 'scipy'],
    output_component_file='new_yamls/split_train_test.yaml')
def get_train_test_split(df_path: Input[Dataset], label_column: str, test_size: float, n_tries: int,
                        output_x_train: Output[Dataset], output_x_test: Output[Dataset],
                        output_y_train: Output[Artifact], output_y_test: Output[Artifact],
                        divergence_output_dict: Output[Artifact]):
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pickle
    import pandas as pd
    import scipy

    def get_kolmogorov_smiron(train_df, test_df, col_name):
        train_df[col_name] = train_df.groupby('analysis_neighborhood', group_keys = False)[col_name].apply(lambda x: x.fillna(x.mean()))
        test_df[col_name] = test_df.groupby('analysis_neighborhood', group_keys = False)[col_name].apply(lambda x: x.fillna(x.mean()))
        statistic = scipy.stats.ks_2samp(train_df[col_name], test_df[col_name]).statistic
        return statistic

    df = pd.read_csv(df_path.path)
    Y = df.pop(label_column)
    cont_cols = ['diff', 'till_game_time', 'latitude', 'longitude']
    results = []
    for random_state in range(n_tries):
        x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size = test_size, random_state = random_state)
        distances = dict(map(lambda a: (a, get_kolmogorov_smiron(x_train, x_test, a)), cont_cols))
        results.append((random_state, distances))

    ks_best = min(results, key=lambda x: np.mean(list(x[1].values())))
    best_seed = ks_best[0]
    ks_dict = ks_best[1]
    x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size = test_size, random_state = best_seed)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train.to_csv(output_x_train.path, index=False)
    x_test.to_csv(output_x_test.path, index=False)
    with open(output_y_train.path, 'wb') as file:
        pickle.dump(y_train, file)
    with open(output_y_test.path, 'wb') as file:
        pickle.dump(y_test, file)
    with open(divergence_output_dict.path, 'wb') as file:
        pickle.dump(ks_dict, file)


@component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas', 'numpy'],
    output_component_file='new_yamls/prep_data_for_training.yaml')
def prepare_data_for_training(x_train_input_path: Input[Dataset], x_test_input_path: Input[Dataset],
                              float_cols: list, cat_cols: list, binary_cols: list,
                              x_train_output_path: Output[Dataset], x_test_output_path: Output[Dataset],
                              cat_label_en_dict_out: Output[Artifact]):

    import pandas as pd
    import pickle
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    req_columns = ['incident_id', 'report_datetime', 'incident_datetime', 'latitude',
                   'longitude', 'datetime', 'incident_date', 'year',
                   'incident_time', 'gameday', 'gametime', 'data_type']
    cat_lab_encoder_dict = {}
    x_train = pd.read_csv(x_train_input_path.path)
    x_test = pd.read_csv(x_test_input_path.path)

    X_TRAIN = pd.DataFrame(x_train[binary_cols + float_cols + req_columns])
    X_TEST = pd.DataFrame(x_test[binary_cols + float_cols + req_columns])


    for col in cat_cols:
        x_train[col] = x_train[col].astype(str)
        x_test[col] = x_test[col].astype(str)
        label_enc = LabelEncoder()
        unique_vals = np.append(x_train[col].unique(), 'UNK')
        x_test.loc[~x_test[col].isin(unique_vals), col] = 'UNK'
        label_enc.fit(unique_vals)
        X_TRAIN[col] = label_enc.transform(x_train[col])
        X_TEST[col] = label_enc.transform(x_test[col])
        cat_lab_encoder_dict[col] = label_enc

    with open(cat_label_en_dict_out.path, 'wb') as file:
        pickle.dump(cat_lab_encoder_dict, file)

    X_TRAIN.to_csv(x_train_output_path.path, index=False)
    X_TEST.to_csv(x_test_output_path.path, index=False)

@component(
    base_image='python:3.9',
    packages_to_install=['xgboost', 'pandas', 'scikit-learn', 'numpy', 'google-cloud-storage'],
    output_component_file='new_yamls/training.yaml')
def train_model(x_train_input_path: Input[Dataset], y_train_input_path: Input[Artifact],
                y_test_input_path: Input[Artifact], y_test_output_path: Output[Artifact],
                y_train_output_path: Output[Artifact], model_output: Output[Artifact],
                label_encoder_path: Output[Artifact], model_dict_path: Output[Artifact],
                output_bucket: str, output_blob_name: str, projectid: str):
    from xgboost import XGBClassifier
    import pandas as pd
    import pickle
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    from datetime import datetime
    from google.cloud import storage

    model_dict = {}
    req_columns = ['incident_id', 'report_datetime', 'incident_datetime', 'latitude',
                   'longitude', 'datetime', 'incident_date', 'year',
                   'incident_time', 'gameday', 'gametime', 'data_type']
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    client = storage.Client(project=projectid)
    bucket = client.get_bucket(output_bucket)
    output_blob_name = f'models/{output_blob_name}_{now}'
    blob = bucket.blob(f'{output_blob_name}/model.pkl')
    gcs_folder_path = f'gs://{output_bucket}/{output_blob_name}'

    x_train = pd.read_csv(x_train_input_path.path)
    with open(y_train_input_path.path, 'rb') as file:
        y_train = pickle.load(file)

    with open(y_test_input_path.path, 'rb') as file:
        y_test = pickle.load(file)

    label_enc_target = LabelEncoder()
    y_train = np.asarray(label_enc_target.fit_transform(y_train))
    y_test = np.asarray(label_enc_target.transform(y_test))

    with open(y_train_output_path.path, 'wb') as file:
        pickle.dump(y_train, file)

    with open(y_test_output_path.path, 'wb') as file:
        pickle.dump(y_test, file)

    with open(label_encoder_path.path, 'wb') as file:
        pickle.dump(label_enc_target, file)

    model = XGBClassifier(learning_rate = 0.03, tree_method='hist', n_estimators = 3,
                          objective='multi:softmax')
    clf = model.fit(x_train.drop(req_columns, axis = 1), y_train)

    model_dict['create_time'] = now
    model_dict['uri'] = gcs_folder_path

    clf.save_model(model_output.path)

    with open(model_dict_path.path, 'wb') as file:
        pickle.dump(model_dict, file)

    blob.upload_from_filename(model_output.path)

@component(
    base_image='python:3.9',
    packages_to_install=['xgboost', 'pandas', 'scikit-learn', 'numpy'],
    output_component_file='new_yamls/predict.yaml')
    # base_image='python:3.10.2')
def predict(x_train_input_path: Input[Dataset], x_test_input_path: Input[Dataset],
            y_train_input_path: Input[Dataset], y_test_input_path: Input[Artifact],
            model_path: Input[Artifact], float_cols: list, bq_data_path: Output[Dataset],
            eval_metrics: Output[Artifact], stats_dict_output: Output[Artifact]):
    import pandas as pd
    import pickle
    import numpy as np
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(model_path.path)
    req_columns = ['incident_id', 'report_datetime', 'incident_datetime', 'latitude',
                   'longitude', 'datetime', 'incident_date', 'year',
                   'incident_time', 'gameday', 'gametime', 'data_type']

    metrics = {}
    x_train = pd.read_csv(x_train_input_path.path)
    x_test = pd.read_csv(x_test_input_path.path)
    with open(y_train_input_path.path, 'rb') as file:
        y_train = pickle.load(file)
    with open(y_test_input_path.path, 'rb') as file:
        y_test = pickle.load(file)

    y_pred_train = model.predict(x_train.drop(req_columns, axis = 1))
    y_pred_test = model.predict(x_test.drop(req_columns, axis = 1))

    stats_dict = x_train[float_cols].describe().to_dict()

    x_train['incident_category'] = y_train
    x_train['Prediction'] = y_pred_train
    x_test['incident_category'] = y_test
    x_test['Prediction'] = y_pred_test

    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    metrics['train_accuracy'] = train_accuracy
    metrics['test_accuracy'] = test_accuracy

    bq_data = pd.concat([x_train, x_test], axis = 0)
    bq_data = bq_data.query('data_type == "ingest"')
    if bq_data.shape[0] > 0:
        bq_data['data_type'] = 'non_ingest'
    bq_data.to_csv(bq_data_path.path, index=False)
    with open(eval_metrics.path, 'wb') as file:
        pickle.dump(metrics, file)
    with open(stats_dict_output.path, 'wb') as file:
        pickle.dump(stats_dict, file)

@component(
    base_image='python:3.9',
    packages_to_install= ['google-cloud-aiplatform', 'xgboost', 'scikit-learn',
                          'google-cloud-storage', 'pandas', 'pyarrow'],
    output_component_file= 'new_yamls/upload_model_registry.yaml')
    # base_image='python:3.10.2')
def upload_model_registry(train_df_path: Input[Artifact], model_path: Input[Artifact], eval_metrics: Input[Artifact],
                          ks_dict_path: Input[Artifact], model_dict_path: Input[Artifact], projectid: str,
                          region: str, accuracy_threshold:float, model_name: str, bucket_name: str,
                          stats_dict_input:Input[Artifact], eval_output: Output[Metrics],
                          output_train_name: str):
    from google.cloud import aiplatform, storage
    import pickle
    import os
    from xgboost import XGBClassifier
    import pandas as pd
    aiplatform.init(project=projectid, location=region)

    with open(eval_metrics.path, 'rb') as file:
        accuracy = pickle.load(file)
        test_accuracy = accuracy['test_accuracy']
        train_accuracy = accuracy['train_accuracy']

    eval_output.log_metric('train accuracy', train_accuracy)
    eval_output.log_metric('test accuracy', test_accuracy)
    with open(model_dict_path.path, 'rb') as file:
        model_dict = pickle.load(file)

    train_df = pd.read_csv(train_df_path.path)
    train_df.to_parquet(output_train_name, compression='gzip')
    client = storage.Client(project=projectid)
    uri = model_dict['uri']
    print (uri)
    uri_folder = '/'.join(uri.split('/')[-2:])
    print ('uri folder: ', uri_folder)
    bucket = client.get_bucket(bucket_name)
    train_df_blob = bucket.blob(f'{uri_folder}/{output_train_name}')
    drift_blob = bucket.blob(f'{uri_folder}/data_drift_dict.pkl')
    metrics_blob = bucket.blob(f'{uri_folder}/metrics_dict.pkl')
    stats_dict_blob = bucket.blob(f'{uri_folder}/stats_dict.pkl')
    model_param_blob = bucket.blob(f'{uri_folder}/model_params.pkl')
    container_image = 'us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest'

    if test_accuracy > accuracy_threshold:
        print ('Model Performance satisfactory: Saving')
        model = XGBClassifier()
        model.load_model(model_path.path)
        params = model.get_params()
        present_models = aiplatform.Model.list(order_by= 'update_time', location= region,)
        print (present_models)
        if len(present_models) > 0:
            previous_model = present_models[-1].resource_name
            uploaded_model = aiplatform.Model.upload(
                                    display_name = model_name,
                                    artifact_uri = uri,
                                    serving_container_image_uri= container_image,
                                    parent_model = previous_model,
                                    is_default_version = True)
        else:
            uploaded_model = aiplatform.Model.upload(
                                    display_name = model_name,
                                    artifact_uri = uri,
                                    serving_container_image_uri= container_image)
        with open('model_params.pkl', 'wb') as file:
            pickle.dump(params, file)
        drift_blob.upload_from_filename(ks_dict_path.path)
        metrics_blob.upload_from_filename(eval_metrics.path)
        stats_dict_blob.upload_from_filename(stats_dict_input.path)
        model_param_blob.upload_from_filename('model_params.pkl')
        train_df_blob.upload_from_filename(output_train_name)


    else:
        print ('Model Performance not satisfactory: Not saving')

@component(
    base_image='python:3.9',
    packages_to_install= ['google-cloud-bigquery'],
    output_component_file= 'new_yamls/delete_bq_data.yaml'
)
def delete_bq_data(df: Input[Dataset], projectid: str, datasetid: str, tableid: str):
    from google.cloud import bigquery as bq

    client = bq.Client(project=projectid)
    q = f'DELETE from {datasetid}.{tableid} where data_type = "ingest"'
    job = client.query(q)
    job.result()

@dsl.pipeline(
    name='SF Training Pipeline',
    description= 'Postprocessing data, Training the model and uploading to registry')

def sf_training_pipeline(projectid:str = projectid, datasetid:str = datasetid, tablename:str = tablename,
                        label_name:str = label_name, test_size:float =test_size, best_split_tries:int =best_split_tries,
                        float_cols:list = float_cols, cat_cols:list = cat_cols, binary_cols:list = binary_cols,
                        bucket_name:str = bucket_name, accuracy_threshold:float = accuracy_threshold,
                        model_name:str = model_name):
    get_data_task = read_training_data(url = '', method='training', projectid=projectid, datasetid=datasetid,
                                       tablename=tablename)
    split_data_task = get_train_test_split(df_path = get_data_task.outputs['output_csv'], label_column = label_name,
                                           test_size = test_size, n_tries=best_split_tries)
    post_proc_train_task = post_processing_after_split(input_df_path= split_data_task.outputs['output_x_train'])

    post_proc_test_task = post_processing_after_split(input_df_path= split_data_task.outputs['output_x_test'])

    data_prep_for_train_task = prepare_data_for_training(x_train_input_path= post_proc_train_task.outputs['output_df_path'],
                              x_test_input_path= post_proc_test_task.outputs['output_df_path'],
                              float_cols= float_cols,
                              cat_cols= cat_cols,
                              binary_cols= binary_cols)

    upload_cat_lab_enc_task = upload_artifact(artifact_path= data_prep_for_train_task.outputs['cat_label_en_dict_out'],
                                              projectid= projectid, bucket_name= bucket_name,
                                              output_path= 'cat_lab_encoder.pkl')

    train_task = train_model(x_train_input_path= data_prep_for_train_task.outputs['x_train_output_path'],
                             y_train_input_path= split_data_task.outputs['output_y_train'],
                             y_test_input_path = split_data_task.outputs['output_y_test'],
                             output_bucket= bucket_name, output_blob_name= 'models',
                             projectid= projectid)

    upload_target_label_enc = upload_artifact(artifact_path= train_task.outputs['label_encoder_path'],
                                              projectid= projectid, bucket_name= bucket_name,
                                              output_path= 'target_lab_encoder.pkl')

    predict_task = predict(x_train_input_path= data_prep_for_train_task.outputs['x_train_output_path'],
                           y_train_input_path= train_task.outputs['y_train_output_path'],
                           model_path= train_task.outputs['model_output'],
                           x_test_input_path= data_prep_for_train_task.outputs['x_test_output_path'],
                           y_test_input_path= train_task.outputs['y_test_output_path'],
                           float_cols = float_cols)

    categ_decode = decode_cols(df_path= predict_task.outputs['bq_data_path'], projectid='ml-deployments',
                               bucket_name= bucket_name, cat_enc_path= 'cat_lab_encoder.pkl',
                               label_enc_path= 'target_lab_encoder.pkl', target_col = label_name,
                               cols_to_decode= cat_cols)

    upload_to_registry_task = upload_model_registry(train_df_path= data_prep_for_train_task.outputs['x_train_output_path'],
                          model_path= train_task.outputs['model_output'], 
                          stats_dict_input = predict_task.outputs['stats_dict_output'],
                          eval_metrics= predict_task.outputs['eval_metrics'],
                          model_dict_path= train_task.outputs['model_dict_path'], projectid= projectid,
                          region= 'us-central1', accuracy_threshold= accuracy_threshold, model_name= model_name,
                          ks_dict_path= split_data_task.outputs['divergence_output_dict'],
                          bucket_name=bucket_name, output_train_name= 'train_df.parquet.gzip')

    delete_duplicate_task= delete_bq_data(df = categ_decode.outputs['output_df'], projectid= projectid, 
                                          datasetid= datasetid, tableid= tablename)

    upload_task = upload_to_bq(df_path = categ_decode.outputs['output_df'], projectid= projectid,
                                datasetid= datasetid, tableid= tablename, location='US',
                                schema = schema)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow pipeline to run ingestion script for SF Crime data.')
    parser.add_argument('type', help='Enter "yaml" if file needs to be saved or "run" if pipline needs to be run directly',
                       type=str)
    parser.add_argument('--test_size', type=float, help='Value for size of the test set, default=0.3')
    parser.add_argument('--best_split_tries', type=int, help='Number of attempts at splitting the data to get the most similar distributions between train and test, default=5')
    parser.add_argument('--accuracy_threshold', type=float, help='Minimum test set accuracy required to push the new model to registry, default=0.3')
    args = parser.parse_args()
    if args.type == 'yaml':
        if not os.path.exists('new_yamls'):
            os.mkdir('new_yamls')
        kfp.compiler.Compiler().compile(
            pipeline_func=sf_training_pipeline,
            package_path='new_yamls/training_pipeline.yaml')
    elif args.type == 'run':
        print ('Here')
        args = {key: value for (key,value) in vars(args).items() if (value != None and key != 'type')}
        print (args)
        client = Client(host= host_url)
        experiments = client.list_experiments().experiments
        experiment_names = list(map(lambda x: x.display_name, experiments))
        if training_experiment_name not in experiment_names:
            client.create_experiment(name=training_experiment_name)
        client.create_run_from_pipeline_func(
        pipeline_func=sf_training_pipeline,
        experiment_name=training_experiment_name,
        arguments=args)