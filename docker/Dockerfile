FROM wenbowen123/foundationpose

RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN conda activate my &&\
    pip install PyOpenGL-accelerate

RUN git clone https://github.com/NVlabs/FoundationPose.git /FoundationPose

RUN cd /FoundationPose/mycpp/ && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j4

RUN conda activate my && cd /kaolin && \
    rm -rf build *egg* && \
    pip install -e .

RUN conda activate my && cd /FoundationPose/bundlesdf/mycuda && \
    rm -rf build *egg* && \
    pip install -e .

RUN apt-get update && apt install -y curl

RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null

RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo focal main" | \
    tee /etc/apt/sources.list.d/librealsense.list > /dev/null
RUN apt-get update

RUN apt-get install -y librealsense2-dkms 
RUN apt-get install -y librealsense2-utils
RUN apt-get install -y librealsense2-dev
RUN apt-get install -y librealsense2-dbg

RUN conda activate my && pip install pyrealsense2

RUN conda activate my && pip install git+https://github.com/facebookresearch/segment-anything.git
RUN conda activate my && pip install pycocotools matplotlib onnxruntime onnx

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh