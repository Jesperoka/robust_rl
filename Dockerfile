FROM ubuntu
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Connect display
WORKDIR /
ENV DISPLAY=host.docker.internal:0.0
ENV make="make -j $(nproc)"
ENV python="python3"

# Apt Install stuff
WORKDIR /
RUN apt-get update
RUN apt-get upgrade
RUN apt-get install -y \
    sudo \
    wget \
    git \
    cmake \
    gcc g++ \
    libpoco-dev \
    libeigen3-dev \
    xorg-dev \
    libusb-dev \
    x11-apps \
    python3.12 \
    python3-pip \
    pybind11-dev \
    catch2 \
    build-essential \
    python3-dev \
    python3-tk \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    v4l-utils \
    gettext \
    udev

# Make python 3.12 the only python
RUN rm -f /usr/bin/python3
RUN ln -s /usr/bin/python3.12 /usr/bin/python3
RUN pip install --upgrade pip --break-system-packages
RUN pip install pybind11 --break-system-packages

# LIBFRANKA
WORKDIR /
RUN git clone --recursive --branch 0.9.2 https://github.com/frankaemika/libfranka
RUN cd libfranka\
 && mkdir build\
 && cd build\
 && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..\
 && cmake --build .\
 && cpack -G DEB \
 && dpkg -i libfranka*.deb

# panda-python 0.7.5 for libfranka 0.9.2 and python 3.11 (which we hope works for python 3.12 as well)
WORKDIR /
RUN git clone https://github.com/JeanElsner/panda-py.git
RUN cd panda-py\
 && rm pyproject.toml
COPY pyproject.toml panda-py/pyproject.toml
RUN cd panda-py\
 && pip install scikit-build --break-system-packages\
 && pip install . --break-system-packages
RUN rm -rf panda-py
WORKDIR /

# Librealsense (built from souce for python 3.12)
# Note: disabling easylogging will break build. Might work on development branch though.
WORKDIR /
RUN git clone https://github.com/IntelRealSense/librealsense.git
RUN cd librealsense \
 && ./scripts/setup_udev_rules.sh \
 && mkdir build \
 && cd build \
 && cmake ../ \
 -DBUILD_SHARED_LIBS=false \
 -DBUILD_PYTHON_BINDINGS:bool=true \
 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
 -DBUILD_EXAMPLES=false \
 -DBUILD_TOOLS=false \
 -DBUILD_UNIT_TESTS=false \
 -DFORCE_RSUSB_BACKEND=true \
 -DBUILD_GLSL_EXTENSIONS=false \
 -DBUILD_WITH_TM2=false \
 -DBUILD_EASYLOGGINGPP=true \
 -DCMAKE_BUILD_TYPE="Release" \
 -DHWM_OVER_XU=false \
 && make uninstall \
 && make clean \
 && make \
 && make install
ENV PYTHONPATH=$PYTHONPATH:/librealsense/build/Release
WORKDIR /

# Pip install jax (Lab PC does not have a GPU)
RUN pip install --upgrade --break-system-packages "jax[cpu]"

# Other python dependencies
RUN pip install --upgrade --break-system-packages \
    numpy \
    numba \
    matplotlib \
    opencv-python \
    pupil-apriltags \
    mujoco-mjx \
    orbax

# Neovim for development (remove this if you're not planning on using neovim to develop on the lab computer)
WORKDIR /
RUN git clone https://github.com/neovim/neovim
RUN cd neovim\
 && make CMAKE_BUILD_TYPE=RelWithDebInfo \
 && make install

WORKDIR /
RUN apt install ripgrep
RUN cd ~ \
 && mkdir .config \
 && cd .config \
 && git clone https://github.com/ThePrimeagen/neovimrc.git \
 && mv neovimrc/ nvim/ \
 && cd /root \
 && mkdir personal \
 && cd personal \
 && git clone https://github.com/ThePrimeagen/harpoon.git \
 && cd harpoon \
 && git switch harpoon2









