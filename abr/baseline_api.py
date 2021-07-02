import bisect
import json
import time
from pathlib import Path

import numpy as np

VIDEO_BITRATE = [400, 800, 1200, 2400, 4800]  # Kbps
REBUFFER_PENALTY = 4.3
SMOOTH_PENALTY = 1
M_IN_K = 1000.0


class BaseAdaptor:
    def __init__(self, log_file: str):
        self.log_path = Path(log_file)
        if self.log_path.exists():
            self.log_path.unlink()

        self.start_time = time.perf_counter()
        self.last_bitrate = 0
        self.last_buffer_size = 0.0

    def write_log(self, buffer_size, file_size, download_rate, quality):
        buffer_size = buffer_size / 1.0  # ms
        video_chunk_size = file_size  # byte
        delay = file_size / (download_rate / 8) * 1000  # ms， from gpac (filesize / download_rate)
        rebuf = np.maximum(delay - self.last_buffer_size, 0.0)  # ms

        bitrate = VIDEO_BITRATE[quality]
        reward = bitrate / M_IN_K \
                 - REBUFFER_PENALTY * rebuf / M_IN_K \
                 - SMOOTH_PENALTY * np.abs(bitrate - self.last_bitrate) / M_IN_K

        self.last_bitrate = bitrate
        self.last_buffer_size = buffer_size

        time_stamp = time.perf_counter() - self.start_time
        line = (str(time_stamp) + '\t' +
                str(bitrate) + '\t' +
                str(buffer_size / M_IN_K) + '\t' +
                str(rebuf / M_IN_K) + '\t' +
                str(video_chunk_size) + '\t' +
                str(delay) + '\t' +
                str(reward) + '\n')

        log_file = self.log_path.open('a', encoding='utf8')
        log_file.write(line)
        log_file.close()


class FixedRateAdaptor(BaseAdaptor):
    def __init__(self, log_file: str):
        super().__init__(log_file)
        self.default_quality = 0

    def select_action(self, buffer_size, file_size, download_rate) -> int:
        quality = self.default_quality

        self.write_log(buffer_size, file_size, download_rate, quality)
        return quality


class BufferBaseAdaptor(BaseAdaptor):
    def __init__(self, log_file: str):
        super().__init__(log_file)
        self.reservoir = 5
        self.cushion = 10

    def select_action(self, buffer_size, file_size, download_rate) -> int:
        buffer = buffer_size / 1000.0  # second
        if buffer < self.reservoir:
            bitrate = VIDEO_BITRATE[0]
        elif buffer > self.reservoir + self.cushion:
            bitrate = VIDEO_BITRATE[4]
        else:
            bitrate = VIDEO_BITRATE[0] + (VIDEO_BITRATE[4] - VIDEO_BITRATE[0]) * (
                    buffer - self.reservoir) / self.cushion

        quality = bisect.bisect_right(VIDEO_BITRATE, bitrate) - 1
        quality = max(quality, 0)

        self.write_log(buffer_size, file_size, download_rate, quality)
        return quality


class RateBaseAdaptor(BaseAdaptor):
    def __init__(self, log_file: str):
        super().__init__(log_file)
        self.p_rb = 1
        self.horizon = 1
        self.past_throughput = []
        self.past_download_time = []

    def predict_throughput(self):
        tmp_sum = 0
        tmp_time = 0
        for throughput, download_time in zip(self.past_throughput[-self.horizon:],
                                             self.past_download_time[-self.horizon:]):
            tmp_sum += download_time / throughput
            tmp_time += download_time
        return tmp_time / tmp_sum

    def select_action(self, buffer_size, file_size, download_rate) -> int:
        self.past_throughput.append(download_rate / 1000.0)
        self.past_download_time.append(file_size / (download_rate / 8.0))

        bitrate = self.p_rb * self.predict_throughput()
        quality = bisect.bisect_right(VIDEO_BITRATE, bitrate) - 1
        quality = max(quality, 0)

        self.write_log(buffer_size, file_size, download_rate, quality)
        return quality
