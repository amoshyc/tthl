import json
from pathlib import Path
from random import randint
from moviepy.editor import VideoFileClip # requires ffmpeg

# dataset/
#     readme.rst
#     LICENSE.md
#     video00/
#         info.json
#         video.mp4
#     video01/
#         info.json
#         video.mp4
#     ...

def process(video_dir):
    """Generating frames and labels of the video.
    Frames is written as `video_dir/%08d.jpg`
    Lables is written back to `info.json`
    
    Arguments:
    video_dir:  A pathlib.Path object pointing to the folder of the target video. 
                The directory should contain `video.mp4` and `info.json`.
    """

    video_path = video_dir / 'video.mp4'
    frame_dir = video_dir / 'frames/'
    frame_dir.mkdir(exist_ok=True)
    frame_fmt = str(frame_dir / '%08d.jpg')
    info_path = video_dir / 'info.json'

    video = VideoFileClip(str(video_path))
    info = json.load(info_path.open())

    print('Generating frames')
    video.write_images_sequence(frame_fmt)

    print('Generating label')
    n_frames = round(video.duration * video.fps)
    info['label'] = [0 for _ in range(n_frames)]
    for s, e in zip(info['starts'], info['ends']):
        fs = round(s * video.fps)
        fe = round(e * video.fps)
        for i in range(fs, fe + 1):
            info['label'][i] = 1
    
    with info_path.open('w') as f:
        json.dump(info, f, ensure_ascii=False)

def main():
    dataset_dir = Path('~/dataset').expanduser().absolute()
    video_dirs = [x for x in dataset_dir.iterdir() if x.is_dir()]
    for i, video_dir in enumerate(video_dirs):
        print(video_dir, '({}/{})'.format(i + 1, len(video_dirs)))
        process(video_dir)
        print('*' * 50)
    
if __name__ == '__main__':
    main()
    