FROM nvcr.io/nvidia/tensorflow:19.09-py3

# Run the copied file and install some dependencies
RUN apt update -qq && \
    apt install --no-install-recommends -y \
    # requirements for LightGBM
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    cmake \
    # requirements for numpy
    libopenblas-base \
    # requirements for keras
    python3-yaml \
    # requirements for pydot
    python3-pydot && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip --no-cache-dir install --upgrade \
    cython \
    h5py \
    scipy \
    keras \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    urllib3 \
    tqdm \
    xgboost \
    catboost \
    fire \
    "dask[complete]" \
    dask-ml \
    tables \
    category_encoders \
    deap update_checker tqdm stopit \
    scikit-mdr skrebate \
    tpot && \
    # installing LightGBM
    git clone --recursive https://github.com/Microsoft/LightGBM && \
    cd LightGBM/python-package && \
    python3 -m pip --no-cache-dir install lightgbm --install-option=--gpu \
        --install-option="--opencl-include-dir=/usr/local/cuda/include/" \
        --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"

# Fix for Error: No OpenCL Device Found
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

ENV CUDA_VISIBLE_DEVICES 0