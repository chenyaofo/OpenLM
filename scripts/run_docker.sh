docker run --gpus all -it --ipc=host \
    -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
    -v /home/chenyaofo:/home/chenyaofo \
    registry.cn-hongkong.aliyuncs.com/chenyaofo/pytorch:2.0-cu118-py310 bash

# -u $(id -u):$(id -g) \