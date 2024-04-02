import logging
import boto3
from botocore.exceptions import ClientError
import os

from core.config import option

s3_client = boto3.client('s3')
def download_file(user_id, video_id, file_name):
    bucket_name = option.download_bucket
    key = f'{user_id}/{video_id}_{file_name}'
    filename = f'{option.source_directory}/{video_id}_{file_name}'

    print(f'bucket name: {bucket_name} key name: {key} filename: {filename} Donwload Start')
    response = s3_client.download_file(Bucket=bucket_name, Key=key
                            , Filename=filename)

    print(f'Donwload End {response}')

    #test-Kim_Seonwoo/0_공항.mov

    # test-Kim_Seonwoo/0공항.mov 원본
    # test-Kim_Seonwoo/0공항.mov 사본

# 4YSA4YWp4Ya84YSS4YWh4Ya8 S3 문자 원본
# 6rO17ZWt [ 텍스트 타이핑 및 내 파일에서 추출한 것 ]

def upload_file(user_id, video_id, file_name):

    print(f'Uploading {file_name} Start')
    # 파일 업로드

    key = f'{user_id}/{video_id}_{file_name}'
    s3_client.delete_object(Bucket=f'{option.result_upload_bucket}', Key=key) # 파일 삭제를 진행함
    print(f'원본 파일 삭제 from s3 key {key}')

    try:
        response = s3_client.upload_file(file_name, option.result_upload_bucket, f'{user_id}/\'{video_id}_{file_name}\'')
        print(f'Response: {response}')
        return True
    except ClientError as e:
        logging.error(e)
        return False