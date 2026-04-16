# Standalone multiarch image that pre-installs TorchBench, HuggingFace, and timm
# dependencies into /opt/conda/envs/py_3.10 so they can be COPY'd into CI images.
#
# Builds for both x86_64 and aarch64.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create jenkins user (matches CI images)
RUN groupadd -g 1014 jenkins && \
    useradd -m -u 1014 -g jenkins jenkins && \
    echo "jenkins ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install miniforge (conda-forge by default, matches CI base images)
ARG PYTHON_VERSION=3.10
ENV ANACONDA_PYTHON_VERSION=${PYTHON_VERSION}
RUN ARCH=$(uname -m) && \
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" && \
    bash /tmp/miniforge.sh -b -f -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda create -y -n py_${PYTHON_VERSION} python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}/bin:/opt/conda/bin:$PATH

# Install pango via conda (matches monolithic build)
RUN conda install -c conda-forge -n py_${ANACONDA_PYTHON_VERSION} -y pango && \
    conda clean -ya

# Pin setuptools<82 (visdom's setup.py needs pkg_resources)
RUN conda run -n py_${ANACONDA_PYTHON_VERSION} pip install --progress-bar off 'setuptools<82'

# Install a temporary CPU-only PyTorch to satisfy TorchBench install checks
RUN conda run -n py_${ANACONDA_PYTHON_VERSION} pip install --progress-bar off \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy pinned commits/requirements
COPY ci_commit_pins/torchbench.txt /pins/torchbench.txt
COPY ci_commit_pins/huggingface-requirements.txt /pins/huggingface-requirements.txt
COPY ci_commit_pins/timm.txt /pins/timm.txt

# Install TorchBench (into / to match monolithic build's CWD)
WORKDIR /
RUN TORCHBENCH_COMMIT=$(cat /pins/torchbench.txt) && \
    git clone https://github.com/pytorch/benchmark torchbench && \
    cd torchbench && \
    git checkout "${TORCHBENCH_COMMIT}" && \
    conda run -n py_${ANACONDA_PYTHON_VERSION} python install.py --continue_on_fail

# Install HuggingFace deps
RUN conda run -n py_${ANACONDA_PYTHON_VERSION} pip install --progress-bar off \
    -r /pins/huggingface-requirements.txt

# Install timm
RUN TIMM_COMMIT=$(cat /pins/timm.txt) && \
    conda run -n py_${ANACONDA_PYTHON_VERSION} pip install --progress-bar off \
    "git+https://github.com/huggingface/pytorch-image-models@${TIMM_COMMIT}"

# Uninstall temporary PyTorch (the real one gets installed in CI)
RUN conda run -n py_${ANACONDA_PYTHON_VERSION} pip uninstall -y \
    torch torchvision torchaudio triton torchao || true

RUN chown -R jenkins:jenkins /torchbench /opt/conda

USER jenkins
CMD ["bash"]
