import configparser

# configparser 객체 생성
config = configparser.ConfigParser()

# .ini 파일 로드
config.read('./config.ini')

print(f'config.ini loaded {config.sections()}')

face_model_path = config['yolo']['face_model_path']
host = config['network']['host']
queue_name = config['network']['queue_name']
source_directory = config['option']['source_directory']

print(f'Source_directory: {source_directory} face model path: {face_model_path} host: {host} queue: {queue_name}')