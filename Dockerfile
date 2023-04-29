FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /code

RUN pip install hydra-core --upgrade
RUN pip install comet-ml pytorch_lightning albumentations omegaconf 
