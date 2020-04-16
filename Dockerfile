FROM ubuntu:18.04

FROM continuumio/miniconda3
ADD environment.yml environment.yml
RUN conda env update -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "decent", "/bin/bash", "-c"]

# Download DNNDK and install it
RUN wget -O xilinx_dnndk_v3.1.tar.gz -nv "https://jug.kplabs.pl/file/cZfqhhaqYz/I53ZXbZyA1"
RUN tar -xf xilinx_dnndk_v3.1.tar.gz && rm -rf xilinx_dnndk_v3.1.tar.gz
RUN pip install xilinx_dnndk_v3.1/host_x86/decent-tf/ubuntu18.04/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
RUN cd xilinx_dnndk_v3.1/host_x86 && ./install.sh
RUN apt-get install -y --force-yes build-essential autoconf libtool libopenblas-dev \
libgflags-dev libgoogle-glog-dev libopencv-dev protobuf-compiler libleveldb-dev \
liblmdb-dev libhdf5-dev libsnappy-dev libboost-all-dev libssl-dev

# Create workspace
RUN mkdir /workspace
WORKDIR /workspace

ADD ml_intuition ml_intuition
ADD scripts scripts
ADD tests tests
VOLUME "/workspace/parameters"

ENV PARAMETERS_DIR "/workspace/parameters"
ENV WORK_DIR "/workspace/work"

ENTRYPOINT ["conda", "run", "-n", "decent"]
