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
    packages_to_install= ['pandas', 'pandas-gbq'],
    output_component_file= 'new_yamls/sf_read_data.yaml')
def read_training_data(url: str, method:str,
                       projectid: str, datasetid: str, tablename: str,
                       output_csv:Output[Dataset]):
    import pandas as pd
    column_mapping = {'Incident Datetime': 'incident_datetime',
                      'Incident Day of Week': 'incident_day_of_week',
                      'Report Datetime': 'report_datetime',
                      'Incident ID': 'incident_id',
                      'Filed Online':'filed_online',
                      'Incident Category': 'incident_category',
                      'Police District': 'police_district',
                      'Analysis Neighborhood': 'analysis_neighborhood',
                      'Supervisor District': 'supervisor_district',
                      'Latitude': 'latitude',
                      'Longitude': 'longitude'}
    if method == 'ingesting':
        df = pd.read_csv(url)
        df.rename(columns=column_mapping, inplace=True)
        df['data_type'] = 'ingest'
    else:
        query = f'SELECT * FROM {projectid}.{datasetid}.{tablename}'
        df = pd.read_gbq(query)
    print (df.shape)
    df.to_csv(output_csv.path, index=False)
    
@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-storage'],
    output_component_file='new_yamls/upload_artifact.yaml')
def upload_artifact(artifact_path: Input[Artifact], projectid: str,
                    bucket_name: str, output_path: str):
    from google.cloud import storage
    import pickle

    storage_client = storage.Client(projectid)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(output_path)
    print (output_path)
    blob.upload_from_filename(artifact_path.path)

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-storage', 'pandas', 'scikit-learn'],
    output_component_file='new_yamls/decode_cols.yaml'
)
def decode_cols(df_path: Input[Dataset], projectid: str, bucket_name: str, target_col: str,
                cat_enc_path: str, label_enc_path: str, cols_to_decode: list, output_df: Output[Dataset]):
    import pandas as pd
    import pickle
    from google.cloud import storage

    df = pd.read_csv(df_path.path)
    storage_client = storage.Client(project=projectid)
    bucket = storage_client.get_bucket(bucket_name)
    cat_enc_blob = bucket.blob(cat_enc_path)
    label_enc_blob = bucket.blob(label_enc_path)
    label_enc_blob.download_to_filename(label_enc_path)
    cat_enc_blob.download_to_filename(cat_enc_path)

    with open(cat_enc_path, 'rb') as file:
        cat_lab_encoder = pickle.load(file)

    with open(label_enc_path, 'rb') as file:
        label_encoder = pickle.load(file)

    for col in cols_to_decode:
        df[col] = cat_lab_encoder[col].inverse_transform(df[col])

    df[target_col] = label_encoder.inverse_transform(df[target_col])
    df.to_csv(output_df.path, index=False)
    
@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy'],
    output_component_file='new_yamls/post_split_processing.yaml'
)
def post_processing_after_split(input_df_path: Input[Dataset], output_df_path: Output[Dataset]):
    import pandas as pd
    import numpy as np

    input_df = pd.read_csv(input_df_path.path)
    cont_cols = ['diff', 'till_game_time', 'latitude', 'longitude']

    for col_name in cont_cols:
        input_df[col_name] = input_df.groupby('analysis_neighborhood', group_keys = False)[col_name].apply(lambda x: x.fillna(x.mean()))

    latitude_buckets = [-np.inf, 37.74, 37.765, 37.795, 37.81, np.inf]
    longitude_buckets = [-np.inf, -122.44, -122.43, -122.39, np.inf]

    input_df['latitude_buckets'] = pd.cut(input_df['latitude'], bins = latitude_buckets,
                                 labels = ['lt_37_74', '37_765', '37_795', '37_81', 'gt_37_81'])
    input_df['longitude_buckets'] = pd.cut(input_df['longitude'], bins = longitude_buckets,
                                 labels = ['lt_neg_122_44', 'neg_122_43', 'neg_122_39', 'gt_neg_122_39'])

    input_df.reset_index(drop=True).to_csv(output_df_path.path)

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-bigquery', 'pandas'],
    output_component_file='new_yamls/upload_bq_comp.yaml'
)
def upload_to_bq(df_path: Input[Dataset], projectid: str, schema: dict,
                 location: str, datasetid: str, tableid: str):

    from google.cloud import bigquery as bq
    import pandas as pd

    df = pd.read_csv(df_path.path)
    if df.shape[0] > 0:
        print (df.columns)
        print (df['year'])
        print (df['year'].dtype)
        schema = [bq.SchemaField(col, format) for col, format in schema.items()]
        client = bq.Client(project=projectid)
        jobconfig = bq.LoadJobConfig(
                    source_format = bq.SourceFormat.CSV, skip_leading_rows=1,
                    schema=schema)
        tablefull = f'{projectid}.{datasetid}.{tableid}'
        dataset_id = "{}.your_dataset".format(client.project)
        try:
            client.get_table(tablefull)
            print ('Table exists: Appending to table')
        except:

            dataset = bq.Dataset(f'{projectid}.{datasetid}')
            dataset.location = location
            dataset = client.create_dataset(dataset, timeout=30)
            print("Created dataset {}.{}".format(client.project, dataset.dataset_id))

            table = bq.Table(tablefull, schema=schema)
            table = client.create_table(table)

        job = client.load_table_from_dataframe(df, tablefull, job_config=jobconfig)
        job.result()
    else:
        print ('No DataFrame to upload')