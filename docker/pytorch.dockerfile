FROM ompl:focal-1.6-mod AS BUILDER

FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as BASE

COPY --from=BUILDER /usr/local/include/ompl-1.6 /usr/include/ompl-1.6
COPY --from=BUILDER /usr/local/lib/libompl* /usr/local/lib/
COPY --from=BUILDER /usr/lib/libtriangle* /usr/lib/
COPY --from=BUILDER /usr/local/bin/ompl_benchmark_statistics.py /usr/bin/ompl_benchmark_statistics.py
COPY --from=BUILDER /usr/lib/python3/dist-packages/ompl /usr/lib/python3/dist-packages/ompl

ENV DEBIAN_FRONTEND=noninteractive

# -----  Files required for OMPL ----------
RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-numpy-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    libboost-test-dev \
    libflann-dev \
    libode-dev \
    libeigen3-dev \
	python3-pip\
	&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
        pypy3 \
        wget && \
    # Install spot
    wget -O /etc/apt/trusted.gpg.d/lrde.gpg https://www.lrde.epita.fr/repo/debian.gpg && \
    echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libspot-dev && \
    pip3 install pygccxml pyplusplus
    
RUN python3 -m pip install -U pip

# -----------------------------------------
RUN pip install torch \
                torchvision \
                torchaudio --index-url https://download.pytorch.org/whl/cu118

# Libgl1 used for open3d
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install einops \
                open3d

# Install torch_geometric
RUN pip install torch_geometric

# Install additional dependencies
RUN pip install pyg_lib \
                torch_scatter \
                torch_sparse \
                torch_cluster \
                torch_spline_conv \
                -f https://data.pyg.org/whl/torch-2.0.0+cu118.html