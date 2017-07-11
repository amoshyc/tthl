import json
from pathlib import Path
from random import randint
from moviepy.editor import VideoFileClip, concatenate_videoclips  # requires ffmpeg

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


def process(video_dir, gen_frames=True, gen_highlight=True, gen_labels=True):
    """Generating frames, highlights and labels of the video.
    Frames is written as `video_dir/%08d.jpg`
    Highlight is generated as `highlight.mp4`
    Labels is written back to `info.json`
    
    Arguments:
    video_dir:  A pathlib.Path object pointing to the folder of the target video. 
                The directory should contain `video.mp4` and `info.json`.
    gen_frames, gen_highlight,  gen_labels: 
                Controls whether to generate frames, highlight, labels repectively.
                Default to True
    """

    video_path = video_dir / 'video.mp4'
    hl_path = video_dir / 'highlight.mp4'
    info_path = video_dir / 'info.json'
    frame_dir = video_dir / 'frames/'
    frame_fmt = frame_dir / '%08d.jpg'
    frame_dir.mkdir(exist_ok=True)

    video = VideoFileClip(str(video_path))
    info = json.load(info_path.open())

    if gen_frames:
        print('Generating frames')
        video.write_images_sequence(str(frame_fmt))

    if gen_highlight:
        print('Generating highlight')
        clips = [video.subclip(s, e) for s, e in zip(info['starts'], info['ends'])]
        hl = concatenate_videoclips(clips)
        hl.write_videofile(str(hl_path), threads=3)

    if gen_labels:
        print('Generating label...', end='')
        n_frames = round(video.duration * video.fps)
        info['label'] = [0 for _ in range(n_frames)]
        for s, e in zip(info['starts'], info['ends']):
            fs = round(s * video.fps)
            fe = round(e * video.fps)
            for i in range(fs, fe + 1):
                info['label'][i] = 1
        with info_path.open('w') as f:
            json.dump(info, f, ensure_ascii=False)
        print('ok')


def main():
    dataset_dir = Path('~/dataset').expanduser().absolute()
    video_dirs = [x for x in dataset_dir.iterdir() if x.is_dir()]
    for i, video_dir in enumerate(video_dirs):
        print(video_dir, '({}/{})'.format(i + 1, len(video_dirs)))
        print()
        process(video_dir)
        print('*' * 50)


if __name__ == '__main__':
    main()
