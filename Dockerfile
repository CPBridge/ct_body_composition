# Start with the tensorflow base image
FROM tensorflow/tensorflow:1.15.0-gpu-py3
MAINTAINER Christopher Bridge

WORKDIR /

# Install useful programs and build tools
RUN apt update && apt install -y \
       build-essential \
       cmake \
       vim \
       swig \
       wget

# Install GDCM from source with python bindings
RUN   wget -O gdcm.tar.gz "https://sourceforge.net/projects/gdcm/files/gdcm 3.x/GDCM 3.0.5/gdcm-3.0.5.tar.gz" && \
      tar xzf gdcm.tar.gz && \
      rm gdcm.tar.gz && \
      mkdir gdcm-build && \
      mkdir /usr/local/lib/python3.6/site-packages && \
      cd gdcm-build && \
      cmake -DGDCM_BUILD_APPLICATIONS=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DGDCM_WRAP_PYTHON=ON \
            -DGDCM_BUILD_SHARED_LIBS=ON \
            -DGDCM_INSTALL_PYTHONMODULE_DIR=/usr/local/lib/python3.6/site-packages \
            -DPYTHON_EXECUTABLE=/usr/bin/python3.6 \
            ../gdcm-3.0.5/ && \
      make -j 16 && \
      make install && \
      ldconfig && \
      rm -rf /gdcm-3.0.5 && \
      rm -rf /gdcm-build

# Clean up packages that are not necessary any more
RUN apt purge -y --auto-remove cmake build-essential && \
    apt clean

# This is necessary so that the GDCM python bindings can be found
ENV PYTHONPATH /usr/local/lib/python3.6/site-packages/

# Install the body composition code
COPY bin /body_comp/bin
COPY body_comp /body_comp/body_comp
COPY setup.py /body_comp/

RUN pip install -e /body_comp/

WORKDIR /body_comp/bin
