import datetime
import json
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os

from core.amqp.rabbit_mq import rabbit_mq_data_producer_to_java_server
from core.config import option

s3_client = boto3.client('s3')


def download_file(user_id, video_id, file_name):
    bucket_name = option.download_bucket
    key = f'{user_id}/{video_id}_{file_name}'
    filename = f'{option.source_directory}/{video_id}_{file_name}'

    print(f'bucket name: {bucket_name} key name: {key} filename: {filename} Download Start')
    response = s3_client.download_file(Bucket=bucket_name, Key=key
                                       , Filename=filename)
    print(f'Download End {response}')


def upload_file(user_id, video_id, result_video_file_path, result_file_name, source_file_name):
    print(f'User ID: {user_id} VideoID: {video_id} result_video_file_path: {result_video_file_path} result_file_name: {result_file_name} source_file_name: {source_file_name}')
    # 파일 업로드

    key = f'{user_id}/{video_id}_{result_file_name}'

    s3_client.delete_object(Bucket=f'{option.download_bucket}', Key=f'{user_id}/{source_file_name}')  # 파일 삭제를 진행함

    #https://mosaic-user-result.s3.ap-northeast-2.amazonaws.com/test-Kim_Seonwoo/0_Result_Result_0_Aiport.mov
    url = f'https://mosaic-user-result.s3.ap-northeast-2.amazonaws.com/{user_id}/{video_id}_{result_file_name}'
    now = datetime.datetime.now()
    message = json.dumps(
        {'date': str(now),
         'type': "RESULT_FILE_URL",
         'url': url}
    )
    rabbit_mq_data_producer_to_java_server.sendToJavaServer(message)
    print(f'원본 파일 삭제 from s3 key {key}')

