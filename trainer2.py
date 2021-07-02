import argparse
import asyncio
import io
import time

import aiohttp
import cv2
import torch
from torch.utils.data import DataLoader

from dataset import OnlineDataset
from process import load_model, training
from selector import Selector

client_id = 'trainer2'
epoch_num = 1
batch_size = 64
fps = 30

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default="127.0.0.1", help='Host for HTTP server (default: 127.0.0.1)')
parser.add_argument('--port', type=int, default=8080, help='Port for HTTP server (default: 8080)')
parser.add_argument('--raw-video', type=str, default='video/raw.mp4', help='Raw video path')
parser.add_argument('--model', type=str, default='model.pth', help='sr model file path')
args = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.set_device(0)

    selector = Selector()
    dataset = OnlineDataset(max_size=2048)

    # emulate high resolution playback
    cap = cv2.VideoCapture(args.raw_video)
    _, frame = cap.read()
    selector.set_last_frame(frame)
    for _ in range(fps - 1):
        _, _ = cap.read()

    _, frame = cap.read()
    selector.set_last_frame(frame)
    for _ in range(fps - 1):
        _, _ = cap.read()
    for scale in range(1, 5):
        selector.scale = scale
        dataset.scale = scale
        patches = selector.select_patches(frame)
        dataset.put_patches(patches)
    loader = DataLoader(dataset=dataset, num_workers=1, persistent_workers=False,
                        batch_size=batch_size, pin_memory=False, shuffle=True)
    model = load_model(args.model)

    async def playing():
        pts = 3
        while cap.isOpened():
            _, frame = cap.read()
            for scale in range(1, 5):
                selector.scale = scale
                dataset.scale = scale
                dataset.put_patches(selector.select_patches(frame))

            for _ in range(fps - 1):
                _, _ = cap.read()

            pts += 1
            await asyncio.sleep(1)


    async def train():
        async with aiohttp.ClientSession() as session:
            url = f'http://{args.host}:{args.port}/ws?client_id={client_id}'
            async with session.ws_connect(url, max_msg_size=8 * 1024 * 1024) as ws:
                while True:
                    msg = await ws.receive()
                    print('recv init state', msg.type, len(msg.data))

                    buffer = io.BytesIO(msg.data)
                    init_state = torch.load(buffer)
                    model.load_state_dict(init_state)

                    pc = time.perf_counter()
                    for scale in range(1, 5):
                        dataset.scale = scale
                        training(model, loader, scale, epoch_num)
                    print(f'training cost {time.perf_counter() - pc}s')

                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    await ws.send_bytes(buffer.getvalue())
                    print('send updated state')

                    await asyncio.sleep(1)


    loop = asyncio.get_event_loop()
    loop.create_task(playing())
    loop.create_task(train())

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.stop()
