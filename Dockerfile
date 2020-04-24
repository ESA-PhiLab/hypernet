ARG cuda_version=10.0

FROM hub.kplabs.pl/cudaconda:${cuda_version}.1-runtime

RUN wget -O cudnn-10.0-linux-x64-v7.4.1.5.tgz -nv https://jug.kplabs.pl/file/kUvED8duLU/iV9OSru55E
RUN tar -xzvf cudnn-10.0-linux-x64-v7.4.1.5.tgz
RUN cp cuda/include/cudnn.h /usr/local/cuda/include
RUN cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
RUN ls /usr/local/cuda/include
RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

RUN ls /usr/local/cuda/include