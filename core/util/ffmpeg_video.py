import ffmpeg


def get_video_properties(video_path):
    # 비디오 파일의 속성 추출
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    print(f'{video_stream} Data')

    # 추출된 속성을 딕셔너리 형태로 저장
    properties = {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps': eval(video_stream['r_frame_rate']),
        'codec': video_stream['codec_name'],
        'format': probe['format']['format_name'],
        'duration': float(video_stream['duration'])
    }
    return properties


def merge_audio(video_input, audio_input, output):
    """
    비디오 파일에 오디오를 병합합니다.

    :param video_input: 처리된 비디오 파일의 경로입니다.
    :param audio_input: 오디오를 추출할 원본 비디오 파일의 경로입니다.
    :param output: 최종 비디오 파일의 저장 경로입니다.
    """
    # 비디오 스트림을 복사하고, 오디오 스트림을 원본 비디오에서 추출하여 합칩니다.
    (
        ffmpeg
        .input(video_input)
        .output(ffmpeg.input(audio_input).audio, output, codec="copy", vcodec="copy")
        .run(overwrite_output=True)
    )
