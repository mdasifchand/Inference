FROM nvcr.io/nvidia/pytorch:22.04-py3

# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:graphics-drivers
RUN apt-get update && apt-get install dialog apt-utils debconf-utils -y
#ARG USERNAME
ARG USER=light
ARG GROUP=developer
ARG UID=1000
ARG GID=1000
RUN apt-get update && \
      apt-get -y install sudo
RUN useradd -m light && echo "light:light" | chpasswd && adduser light sudo && su light
RUN pip install conan && source ~/.profile 
ENV PIP_ROOT_USER_ACTION=ignore
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin
USER light
WORKDIR /home/light
CMD ["bash"]

