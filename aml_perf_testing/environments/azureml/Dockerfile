# Adding NCCL tests on top of AzureML curated environment

# check release notes https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04/release-notes.md
#FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:latest
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu121-py38-torch210:latest

USER root:root
ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    redis \
    iproute2 && \
    # rdma-core dependencies
    apt-get install -y \
    udev \
    libudev-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    libgnutls30 \
    gcc \
    ninja-build \
    pkg-config \
    valgrind \
    cython3 \
    python3-docutils \
    pandoc \
    python3-dev && \
    # Others
    apt-get install -y \
    build-essential \
    bzip2 \
    libbz2-1.0 \
    systemd \
    git \
    wget \
    cpio \
    pciutils \
    libnuma-dev \
    ibutils \
    ibverbs-utils \ 
    rdmacm-utils \
    infiniband-diags \
    perftest \
    librdmacm-dev \
    libibverbs-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libssl1.1 \
    libglib2.0-0 \
    dh-make \
    #libnettle6 \
    libx11-dev \
    libcurl3-nss \
    libgd3 \
    nginx \
    binutils-multiarch \
    # adding below packages to mitigate the vulnerabilities till libpcre3
    libcurl3-nss \
    e2fsprogs \
    e2fsck-static \
    perl \
    libksba8 \
    libkrb5-26-heimdal \
    libhx509-5-heimdal \
    libheimntlm0-heimdal \
    libheimbase1-heimdal \
    libasn1-8-heimdal \
    libroken18-heimdal \
    libwind0-heimdal \
    libgssapi3-heimdal \
    libhcrypto4-heimdal \
    libpam-modules \
    libpam-modules-bin \
    libpam0g \
    libpam-runtime \
    fuse2fs \
    gnupg \
    gnupg2 \
    gpg \
    libdpkg-perl \
    dpkg \
    libpcre3 \ 
    sqlite3 \
    #adding packages to mitigate vulnerabilities
    passwd \
    login \
    uidmap \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*



# Open-MPI-UCX installation
RUN mkdir /tmp/ucx && \
    cd /tmp/ucx && \
        wget -q https://github.com/openucx/ucx/releases/download/v1.9.0/ucx-1.9.0.tar.gz && \
        tar zxf ucx-1.9.0.tar.gz && \
	cd ucx-1.9.0 && \
        ./configure --prefix=/usr/local --enable-optimizations --disable-assertions --disable-params-check --enable-mt && \
        make -j $(nproc --all) && \
        make install && \
        rm -rf /tmp/ucx


# Open-MPI installation
ENV OPENMPI_VERSION 4.1.0
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-ucx=/usr/local/ --enable-mca-no-build=btl-uct --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi	
	


#Cmake Installation
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y dh-python 

# rdma-core v30.0 for Mlnx_ofed_5_1_2 as user space driver
RUN mkdir /tmp/rdma-core && \
    cd /tmp/rdma-core && \
    git clone --branch v30.0 https://github.com/linux-rdma/rdma-core && \
    cd /tmp/rdma-core/rdma-core && \
    debian/rules binary && \
    dpkg -i ../*.deb && \
    rm -rf /tmp/rdma-core

#Install latest version of nccl-rdma-sharp-plugins
RUN cd /tmp && \
    mkdir -p /usr/local/nccl-rdma-sharp-plugins && \
    apt install -y dh-make zlib1g-dev && \
    git clone -b v2.1.0 https://github.com/Mellanox/nccl-rdma-sharp-plugins.git && \
    cd nccl-rdma-sharp-plugins && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda --without-ucx && \
    make && \
    make install
    
# set env var to find nccl rdma plugins inside this container
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl-rdma-sharp-plugins/lib


# set env var to find nccl rdma plugins inside this container
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nccl-rdma-sharp-plugins/lib"

##############################################################################
# NCCL TESTS
##############################################################################
ENV NCCL_TESTS_TAG=v2.11.0

# NOTE: adding gencodes to support K80, M60, V100, A100
RUN mkdir /tmp/nccltests && \
    cd /tmp/nccltests && \
    git clone -b ${NCCL_TESTS_TAG} https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
        make \
            MPI=1 \
            # NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90" \
            NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70" \
            CUDA_HOME=/usr/local/cuda && \
        cp ./build/* /usr/local/bin && \
    rm -rf /tmp/nccltests

# for lstopo
RUN apt update && \
    apt install hwloc -y && \
    rm -rf /var/lib/apt/lists/*
