import argparse
import bisect
import io
import time
from queue import Queue
from typing import List

import cv2
import numpy as np
import requests
import torch
import torch.multiprocessing as mp
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import libgpac as gpac
from abr.api import BitrateAdaptor
from process import load_model, inference

gpac.init()

SAMPLE_INTERVAL = 30 * 2
SAMPLE_NUM = 298
FRAME_TIME = 1 / 30

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host  (default: 127.0.0.1)')
parser.add_argument('--port', type=str, default=8080, help='Server port  (default: 8080)')
parser.add_argument('--raw-video', type=str, default='video/raw.mp4', help='Raw video path')
parser.add_argument('--log-file', type=str, default='qoe_log.txt', help='QoE log path')
parser.add_argument('--model', type=str, default='model.pth', help='sr model file path')
args = parser.parse_args()


class MyCustomDASHAlgo:
    def __init__(self):
        self.count = 0
        self.bitrate_level = 5
        self.adaptor = BitrateAdaptor('video/size/size.json', args.log_file)

    def on_rate_adaptation(self, group, base_group, force_low_complexity, stats):
        self.count += 1
        print(f'rate adaptation on segment {self.count}')
        action = self.adaptor.select_action(stats.buffer, stats.filesize, stats.download_rate, sr_gains)
        if action >= self.bitrate_level:
            quality = action - self.bitrate_level
            scale = (self.bitrate_level - 1) - quality
            r = requests.get(f'http://{args.host}:{args.port}/model', params={'scale': scale})
            model_queue.put((scale, r.content))
        else:
            quality = action

        global buffer_size
        buffer_size = stats.buffer

        return quality


# create an instance of the algo
mydash = MyCustomDASHAlgo()


# define a custom filter session monitoring the creation of new filters
class MyFilterSession(gpac.FilterSession):
    def __init__(self, flags=0, blacklist=None, nb_threads=0, sched_type=0):
        gpac.FilterSession.__init__(self, flags, blacklist, nb_threads, sched_type)

    def on_filter_new(self, f):
        # bind the dashin filter to our algorithm object
        if f.name == 'dashin':
            f.bind(mydash)

    def on_filter_del(self, f):
        print('del filter ' + f.name)


# define a custom filter
class MyFilter(gpac.FilterCustom):
    def __init__(self, session):
        gpac.FilterCustom.__init__(self, session, "PYRawVid")
        # indicate what we accept and produce - here, raw video in and out
        self.push_cap("StreamType", "Visual", gpac.GF_CAPS_INPUT)
        self.push_cap("CodecID", "Raw", gpac.GF_CAPS_INPUT)

        self.frame_count = 0
        self.sample_interval = SAMPLE_INTERVAL

    # configure input pids
    def configure_pid(self, pid, is_remove):
        if is_remove:
            return 0
        if pid in self.ipids:
            pass
        else:
            evt = gpac.FilterEvent(gpac.GF_FEVT_BUFFER_REQ)
            # maximum buffer level in microseconds, here 60 seconds
            evt.buffer_req.max_buffer_us = 60 * 1000 * 1000
            pid.send_event(evt)

            # we are a sink, we MUST fire a play event
            evt = gpac.FilterEvent(gpac.GF_FEVT_PLAY)
            pid.send_event(evt)

        # get width, height, stride and pixel format - get_prop may return None if property is not yet known
        # but this should not happen for these properties with raw video, except StrideUV which is NULL for non (semi) planar YUV formats
        self.width = pid.get_prop('Width')
        self.height = pid.get_prop('Height')
        self.pixfmt = pid.get_prop('PixelFormat')

        return 0

    # process
    def process(self):
        for pid in self.ipids:
            pck = pid.get_packet()
            if pck is None:
                break

            data = pck.data
            yuv = data.reshape((self.height * 3 // 2, self.width))
            if self.pixfmt == 'nv12':
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            else:
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

            if self.frame_count % self.sample_interval == 0:
                frame_queue.put(rgb)
            self.frame_count += 1

            pid.drop_packet()

            # playback
            time.sleep(FRAME_TIME)
        return 0


def calculate_metrics(frame_queue: Queue, model_queue: Queue, sr_gains: List):
    torch.cuda.set_device(0)

    pts = 0
    sample_interval = SAMPLE_INTERVAL
    cap = cv2.VideoCapture(args.raw_video)
    model = load_model(args.model)
    model.eval()

    while True:
        while model_queue.qsize() > 0:
            scale, data = model_queue.get()
            if model_queue.qsize() == 0:
                buffer = io.BytesIO(data)
                model.networks[scale - 1].load_state_dict(torch.load(buffer))

        frame = frame_queue.get(timeout=120)
        bi_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        height = frame.shape[0]
        if height == 1080:
            online_sr_frame = frame
        else:
            if height == 720:
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            scale = int(1080 // height)
            online_sr_frame = inference(model, frame, scale)

        ret, hr_frame = cap.read()
        hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
        for _ in range(sample_interval - 1):
            _, _ = cap.read()

        def calc_metric(metric_name: str, frames: dict, hr_frame: np.ndarray):
            assert metric_name in ['psnr', 'ssim']
            metric = ssim if metric_name == 'ssim' else psnr
            opts = {'multichannel': True} if metric_name == 'ssim' else {}
            return {k: metric(hr_frame, v, **opts) for k, v in frames.items()}

        frames = {
            'bicubic': bi_frame,
            'online': online_sr_frame,
        }

        # psnr_result = calc_metric('psnr', frames, hr_frame)
        ssim_result = calc_metric('ssim', frames, hr_frame)

        def reverse_ssim(val):
            points = [(0, 0), (400, 0.8255304244878122), (800, 0.8703809272401022),
                      (1200, 0.9148781387569929), (2400, 0.9438733564415974), (4800, 1)]
            a = [p[1] for p in points]
            i = bisect.bisect_left(a, val, lo=0, hi=len(a))
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            f = np.poly1d(np.polyfit([y1, y2], [x1, x2], 1))
            return f(val)

        if height != 1080:
            eq_bitrate = reverse_ssim(ssim_result['online'])
            height_quality_map = {270: 0, 360: 1, 540: 2, 720: 3, 1080: 4}
            height_bitrate_map = {270: 400, 360: 800, 540: 1200, 720: 2400, 1080: 4800}
            sr_gains[height_quality_map[height]] = eq_bitrate / height_bitrate_map[height]

        pts += 1
        if pts >= SAMPLE_NUM:
            break

    print('process finished')


if __name__ == '__main__':
    mp.set_start_method('spawn')

    manager = mp.Manager()
    sr_gains = manager.list([1.8257, 2.0375, 1.8616, 1.1783, 1])
    frame_queue = manager.Queue()
    model_queue = manager.Queue()

    process = mp.Process(target=calculate_metrics, args=(frame_queue, model_queue, sr_gains))
    process.start()

    fs = MyFilterSession(0, 'nvdec')

    # load a source filter
    src = fs.load_src(f'http://{args.host}:{args.port}/dash.mpd:gpac:start_with=min_bw')

    # load a custom filter
    my_filter = MyFilter(fs)
    my_filter.set_source(src)

    # and run
    fs.run()
    process.join()
