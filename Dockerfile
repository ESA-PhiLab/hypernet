ARG cuda_version=10.0

FROM hub.kplabs.pl/cudaconda:${cuda_version}.1-runtime

RUN wget -O libcudnn7_7.4.1.5-1+cuda10.0_amd64.deb -nv https://jug.kplabs.pl/file/nrF9cp6TA5/A7lKa0dml0
RUN dpkg -i libcudnn7_7.4.1.5-1+cuda10.0_amd64.deb

RUN ls /usr/src/cudnn_samples_v7/