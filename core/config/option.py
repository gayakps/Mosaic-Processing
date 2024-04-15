import configparser

# configparser 객체 생성
config = configparser.ConfigParser()

# .ini 파일 로드
config.read('./config.ini')

print(f'config.ini loaded {config.sections()}')

face_model_path = config['yolo']['face_model_path']
host = config['network']['host']

queue_name = config['network']['queue_name']
user_name = config['network']['user_name']
password = config['network']['password']

print(f'username: {user_name} password: {password}')

source_directory = config['option']['source_directory']

download_bucket = config['s3']['download_bucket']
result_upload_bucket = config['s3']['result_upload_bucket']

print(f'Source_directory: {source_directory} face model path: {face_model_path} host: {host} queue: {queue_name}')