# LiveSR_public

This is part of the source code of paper LiveSR: Enabling Universal HD Live Streaming with Crowdsourced Online Learning.

It just uses for demonstrating the framework of the system.


## Run
0. Requirements:
    * numpy
    * pytorch
    * libgpac
    * aiohttp
    * requests
    * opencv-python
    * scikit-image


1. Prepare data

    Preparing a high resolution video(1920*1080) and use `generate_data` in `tools.py` to process the video.
    For example, the directory should like below after processing.
    ```
    video
    |--dash
    |--size
    |--source
    |--raw.mp4
    ```

2. Start service

    Run the python scripts and add the option on your need.
    
    `python server.py`
    
    `python client.py`

    `python trainer1.py`
    
    `python trainer2.py`


## Citing

TBD
