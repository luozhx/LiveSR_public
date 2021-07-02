import queue
import random
from typing import List, Tuple

import cv2
import numpy as np


class RandomSelector:
    def __init__(self, max_size=30):
        self.frames = []

        self.max_size = max_size
        self.patch_size = 64
        self._scale = 2

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        assert val in [1, 2, 3, 4]
        self._scale = val

    def put_frame(self, frame: np.ndarray):
        if len(self.frames) > self.max_size:
            self.frames = self.frames[self.max_size // 4:]
        self.frames.append(frame)

    def __select_patches_from_frame(self, frame: np.ndarray, num: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        patch_size = self.patch_size
        scale = self.scale
        height, width, _ = frame.shape
        if scale in [2, 3, 4]:
            height, width = height // scale, width // scale
        elif scale == 1:
            height, width = 720, 1280
        lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        patches = []
        for _ in range(num):
            x = random.randrange(0, width - patch_size + 1)
            y = random.randrange(0, height - patch_size + 1)
            lr_patch = lr[y:y + patch_size, x:x + patch_size]
            hr_patch = frame[y * scale:(y + patch_size) * scale, x * scale:(x + patch_size) * scale]
            patches.append((lr_patch, hr_patch))

        return patches

    def select_patches(self, source_frame_num=1, patch_per_frame=64) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        for _ in range(source_frame_num):
            frame_id = random.randrange(0, len(self.frames))
            frame = self.frames[frame_id]
            patches += self.__select_patches_from_frame(frame, patch_per_frame)
        return patches


class PatchItem:
    def __init__(self, hr: np.ndarray, distance: float):
        self.hr = hr
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance


class Selector:
    def __init__(self):
        self._scale = 2
        self.patch_size = 32
        self.last_frame = None

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        assert val in [1, 2, 3, 4]
        self._scale = val

    def set_last_frame(self, frame):
        self.last_frame = frame

    def select_patches(self, current_frame: np.ndarray, patch_number=64) -> List[Tuple[np.ndarray, np.ndarray]]:
        last_frame = self.last_frame
        height, width, _ = current_frame.shape
        hr_patch_size = self.patch_size * self.scale

        patch_queue = queue.PriorityQueue()
        m, n = width // hr_patch_size, height // hr_patch_size

        for i in range(m):
            for j in range(n):
                x, y = i * hr_patch_size, j * hr_patch_size
                current_roi = current_frame[y: y + hr_patch_size, x:x + hr_patch_size]
                last_roi = last_frame[y: y + hr_patch_size, x:x + hr_patch_size]
                mse = np.mean((current_roi - last_roi) ** 2)
                patch_queue.put(PatchItem(current_roi, float(mse)))

        patches = []
        for i in range(patch_number):
            item = patch_queue.get()
            hr = item.hr
            if self.scale == 1:
                patch_size = int(self.patch_size / 1.5)
                lr = cv2.resize(hr, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                lr = cv2.resize(lr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            else:
                lr = cv2.resize(hr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
            patches.append((lr, hr))

        self.last_frame = current_frame
        return patches
