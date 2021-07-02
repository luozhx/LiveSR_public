import json
import shlex
import subprocess
import sys
from pathlib import Path


def split(cmd):
    return cmd if sys.platform == 'win32' else shlex.split(cmd)


preset = [(1920, 1080, '4800k'), (1280, 720, '2400k'), (960, 540, '1200k'), (640, 360, '800k'), (480, 270, '400k')]


def scale_video(src: Path, dst: Path, width: int, height: int, bitrate: str, fps: int, gop: int) -> None:
    cmd = f'ffmpeg -i {src} -c:v hevc -an -preset slow -r {fps} -s {width}x{height} -b:v {bitrate} -x265-params keyint={gop}:no-open-gop=1 -sws_flags bicubic {dst}'
    r = subprocess.run(split(cmd), capture_output=True, text=True, encoding='utf8')
    if r.returncode != 0:
        print(r.stderr)


def dash_video(src_dir: Path, dst: Path, duration: int = 4000) -> None:
    src = ' '.join([f'../source/{x.name}:id={x.stem}' for x in src_dir.iterdir()])
    cmd = f'MP4Box -dash {duration} -rap -profile live -bs-switching no -segment-name $RepresentationID$/segment_$Number$$Init=init$ -out dash.mpd {src}'
    r = subprocess.run(split(cmd), cwd=dst, capture_output=True, text=True, encoding='utf8')
    if r.returncode != 0:
        print(r.stderr)


def get_video_size(src: Path):
    dash_dir = src / 'dash'
    size_dir = src / 'size'
    size_dir.mkdir(exist_ok=True)

    quality_list = ['270p', '360p', '540p', '720p', '1080p']
    video_sizes = []
    for i, x in enumerate(quality_list):
        segments_sizes = []
        for y in (dash_dir / x).iterdir():
            if y.suffix != '.m4s':
                continue
            segments_sizes.append((int(y.stem[8:]), y.stat().st_size))
        segments_sizes.sort(key=lambda it: it[0])
        segments_sizes = [x for _, x in segments_sizes]
        video_sizes.append(segments_sizes)

        with (size_dir / f'video_size_{i}').open('w', encoding='utf8') as f:
            for size in segments_sizes:
                f.write(str(size) + '\n')

    with (size_dir / f'size.json').open('w', encoding='utf8') as f:
        json.dump(video_sizes, f, indent=4)


def generate_data(src: Path, fps: int, duration: int) -> None:
    for width, height, bitrate in preset:
        dst_dir = Path(src).parent / 'source'
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f'{height}p.mp4'
        scale_video(src, dst, width, height, bitrate, fps, fps * duration)

    dst_dir = Path(src).parent / 'dash'
    dst_dir.mkdir(parents=True, exist_ok=True)
    dash_video(src.parent / 'source', dst_dir, duration * 1000)

    get_video_size(Path(src).parent)
