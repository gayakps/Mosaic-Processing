import logging
import boto3
from botocore.exceptions import ClientError
import os


def upload_file(file_name, bucket, object_name=None):
    """ S3 버킷에 파일을 업로드합니다.

    :param file_name: 업로드할 파일
    :param bucket: 업로드될 버킷
    :param object_name: S3 객체이름. 없으면 file_name 사용
    :return: 파일이 업로드되면 True, 아니면 False
    """

    # S3 객체이름이 정의되지 않으면, file_name을 사용
    if object_name is None:
        object_name = os.path.basename(file_name)

    print(f'Uploading {object_name} Start')

    # 파일 업로드
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
        print(f'Response: {response}')
    except ClientError as e:
        logging.error(e)
        return False
    return True