import ffmpeg


def get_video_properties(video_path):
    # 비디오 파일의 속성 추출
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    # 추출된 속성을 딕셔너리 형태로 저장
    properties = {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps': eval(video_stream['r_frame_rate']),
        'codec': video_stream['codec_name'],
        'format': probe['format']['format_name']
    }
    return properties