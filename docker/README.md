# Docker Image

## Using pre-built images

You can also pull a pre-built docker image from Aliyun Docker Registry

```
docker run --gpus all -it --ipc=host -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro registry.cn-hongkong.aliyuncs.com/chenyaofo/openlm:py310-cu118 bash
```

## Building the image by yourself

You can modify `Dockerfile` by yourself (such as versions of CUDA, PyTorch and etc.) and build it by

```
docker build -t openlm:py310-cu118 -f Dockerfile .
```
