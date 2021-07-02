import argparse
import asyncio
import io
import logging
from dataclasses import dataclass
from typing import List

import torch
from aiohttp import web

from process import load_model


@dataclass
class WsClient:
    client_id: str
    state: dict
    ws: web.WebSocketResponse


client_num: int = 2
updated_client_num: int = 0
clients: List[WsClient] = []


async def get_model(request: web.Request):
    scale = int(request.rel_url.query.get('scale'))
    if scale not in [1, 2, 3, 4]:
        return web.HTTPBadRequest

    buffer = io.BytesIO()
    state_dict = model.networks[scale - 1].state_dict()
    torch.save(state_dict, buffer)
    return web.Response(body=buffer.getvalue())


async def websocket_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)

    client_id = request.rel_url.query.get('client_id')
    client = WsClient(client_id, {}, ws)
    clients.append(client)
    if len(clients) == client_num:
        # start training
        asyncio.create_task(fedavg())

    # loop forever
    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            print(f'receive state from {client_id}')
            buffer = io.BytesIO(msg.data)
            model.load_state_dict(torch.load(buffer))  # load to cpu
            client.state = model.state_dict()

            global updated_client_num
            updated_client_num += 1

    return ws


async def fedavg():
    while True:
        buffer = io.BytesIO()
        torch.save(global_state, buffer)

        ret = await asyncio.gather(
            *[cli.ws.send_bytes(buffer.getvalue()) for cli in clients]
        )

        global updated_client_num
        while updated_client_num != client_num:
            await asyncio.sleep(1)
        updated_client_num = 0

        states = [x.state for x in clients]
        states_diff = []
        for state in states:
            diff = {}
            for key in state:
                diff[key] = state[key] - global_state[key]
            states_diff.append(diff)

        for diff in states_diff:
            for key in global_state:
                global_state[key] += diff[key] / len(states_diff)

        model.load_state_dict(global_state)
        print('global state updated')

        await asyncio.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host for HTTP server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port for HTTP server (default: 8080)')
    parser.add_argument('--dash-video', type=str, default='video/dash', help='Path for DASH video (default: video)')
    parser.add_argument('--model', type=str, default='model.pth', help='sr model file path')
    args = parser.parse_args()

    model = load_model(args.model)
    global_state = model.state_dict()

    app = web.Application()
    app.add_routes([web.get('/ws', websocket_handler)])

    app.router.add_get("/model", get_model)
    app.router.add_static('/', args.dash_video)

    logging.basicConfig(level=logging.DEBUG)
    web.run_app(app, host=args.host, port=args.port)
