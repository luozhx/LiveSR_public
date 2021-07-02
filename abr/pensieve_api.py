import json
import time
from pathlib import Path

import numpy as np
import torch

from abr.A3C import A3C

IS_CENTRAL = False
S_INFO = 6
S_LEN = 8
A_DIM = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
ACTOR_MODEL = 'abr/pensieve_actor.pth'

VIDEO_BIT_RATE = [400, 800, 1200, 2400, 4800]  # Kbps
BITRATE_LEVELS = len(VIDEO_BIT_RATE)
BUFFER_NORM_FACTOR = 10.0
VIDEO_CHUNK_LEN = 4000.0
CHUNK_TIL_VIDEO_END_CAP = 148.0
M_IN_K = 1000.0

REBUFFER_PENALTY = 4.3
SMOOTH_PENALTY = 1


class PensieveAdaptor:
    def __init__(self, video_size: str, log_file: str):
        self.log_file = Path(log_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.segment_pointer = 0
        self.last_quality = 0
        self.last_buffer_size = 0
        self.last_bitrate = VIDEO_BIT_RATE[self.last_quality]
        self.last_state = torch.zeros((1, S_INFO, S_LEN))

        self.net = A3C(IS_CENTRAL, [S_INFO, S_LEN], A_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE)
        self.net.actor.load_state_dict(torch.load(ACTOR_MODEL))

        with Path(video_size).open('r', encoding='utf8') as f:
            self.segment_sizes = json.load(f)

        self.start_time = time.perf_counter()

    def select_action(self, buffer_size, file_size, download_rate) -> int:
        buffer_size = buffer_size / 1000.0  # second,  from gpac (ms)
        video_chunk_size = file_size  # byte,  from gpac or env
        next_video_chunk_sizes = []  # from env
        for i in range(len(VIDEO_BIT_RATE)):
            if self.segment_pointer + 1 < len(self.segment_sizes[i]):
                size = self.segment_sizes[i][self.segment_pointer + 1]
            else:
                size = 0
            next_video_chunk_sizes.append(size)
        video_chunk_remain = max(0.0, CHUNK_TIL_VIDEO_END_CAP - self.segment_pointer)
        delay = file_size / (download_rate / 8) * 1000  # ms， from gpac (filesize / download_rate)
        rebuf = np.maximum(delay / 1000.0 - self.last_buffer_size, 0.0)  # second

        state = torch.roll(self.last_state, -1, dims=-1).detach().to(self.device)
        state[0, 0, -1] = self.last_bitrate / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[0, 1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[0, 2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[0, 3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[0, 4, :len(next_video_chunk_sizes)] = torch.tensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[0, 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        action = self.net.select_action(state)
        quality = action

        # --linear reward--
        bitrate = VIDEO_BIT_RATE[quality]
        reward = bitrate / M_IN_K \
                 - REBUFFER_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(bitrate - self.last_bitrate) / M_IN_K

        self.last_state = state
        self.last_buffer_size = buffer_size
        self.last_quality = quality
        self.last_bitrate = bitrate

        time_stamp = time.perf_counter() - self.start_time
        line = (str(time_stamp) + '\t' +
                str(bitrate) + '\t' +
                str(buffer_size) + '\t' +
                str(rebuf) + '\t' +
                str(video_chunk_size) + '\t' +
                str(delay) + '\t' +
                str(reward) + '\n')

        log_file = self.log_file.open('a', encoding='utf8')
        log_file.write(line)
        log_file.close()

        return action
