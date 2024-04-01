import logging
import boto3
from botocore.exceptions import ClientError
import os

from core.config import option

s3_client = boto3.client('s3')
def download_file(user_id, file_name):
    bucket_name = option.download_bucket
    s3_client.download_file(Bucket=bucket_name, key=f'{user_id}/{file_name}'
                            , filename=f'{option.source_directory}/\'{file_name}\'')


def upload_file(file_name, user_id):

    print(f'Uploading {file_name} Start')
    # 파일 업로드
    s3_client.delete_object(Bucket=f'{option.download_bucket}', Key=f'{user_id}/{file_name}') # 파일 삭제를 진행함
    print('원본 파일 삭제 from s3')

    try:
        response = s3_client.upload_file(file_name, option.result_upload_bucket, f'{user_id}/\'{file_name}\'')
        print(f'Response: {response}')
        return True
    except ClientError as e:
        logging.error(e)
        return False