#FROM nvidia/cuda:10.0-runtime-ubuntu18.04
#FROM nvidia/cuda
#ARG cuda_version=10.0

#FROM hub.kplabs.pl/cudaconda:${cuda_version}.1-runtime

FROM ubuntu:18.04

FROM continuumio/miniconda3
ADD environment.yml environment.yml
RUN conda env update -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "decent", "/bin/bash", "-c"]

RUN apt-get -y update && apt-get -y --force-yes install gnupg

# Install CUDA
RUN wget -O cuda-repo-ubuntu1804_10.0.130-1_amd64.deb -nv "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb"
RUN dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install cuda-10-0

# Install CUDnn
RUN wget -O cudnn-10.0-linux-x64-v7.4.1.5.tgz -nv https://jug.kplabs.pl/file/kUvED8duLU/iV9OSru55E
RUN tar -xzvf cudnn-10.0-linux-x64-v7.4.1.5.tgz
RUN mkdir usr/local/cuda-10.0/include
RUN cp -P cuda/include/cudnn.h usr/local/cuda-10.0/include
RUN cp -P cuda/lib64/libcudnn* usr/local/cuda-10.0/lib64
RUN chmod a+r usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*

# Download DNNDK and install it
RUN wget -O xilinx_dnndk_v3.1.tar.gz -nv "https://jug.kplabs.pl/file/cZfqhhaqYz/I53ZXbZyA1"
RUN tar -xf xilinx_dnndk_v3.1.tar.gz && rm -rf xilinx_dnndk_v3.1.tar.gz
RUN pip install xilinx_dnndk_v3.1/host_x86/decent-tf/ubuntu18.04/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl
#RUN cd xilinx_dnndk_v3.1/host_x86 && ./install.sh
#RUN apt-get -y update && apt-get install -y --force-yes libtool build-essential autoconf libopenblas-dev \
#libgflags-dev libgoogle-glog-dev libopencv-dev protobuf-compiler libleveldb-dev \
#liblmdb-dev libhdf5-dev libsnappy-dev libboost-all-dev libssl-dev
RUN apt-get -y update && apt-get install -y --force-yes libgomp1
#RUN conda install cudatoolkit=10.0
#RUN conda install cudnn=7.4.1.5
#RUN conda list
#RUN echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
#RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

# Create workspace
RUN mkdir /workspace
WORKDIR /workspace

ADD ml_intuition ml_intuition
ADD scripts scripts
ADD tests tests
VOLUME "/workspace/parameters"

ENV PARAMETERS_DIR "/workspace/parameters"
ENV WORK_DIR "/workspace/work"

ENTRYPOINT ["conda", "run", "-n", "decent", "/bin/bash", "-c"]