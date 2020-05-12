FROM alpine:3.11

LABEL MAINTAINER="Max van den Boom <m.a.vandenboom84@gmail.com>"

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# This hack is widely applied to avoid python printing issues in docker containers.
# See: https://github.com/Docker-Hub-frolvlad/docker-alpine-python3/pull/13
ENV PYTHONUNBUFFERED=1

# install python
RUN apk add --no-cache python3=3.8.2-r0 && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    rm -r /root/.cache && \
    cd /usr/local/bin && \
    ln -s idle3 idle && \
    ln -s pydoc3 pydoc && \
    ln -s python3 python && \
    ln -s python3-config python-config

# copy pymef to the image
COPY pymef /pymef

# build and install numpy, scipy, mne, bids_validator and pymef
RUN apk add --no-cache libpng freetype libstdc++ openblas lapack libxml2 libxslt \
	&& apk add --no-cache --virtual .build-deps \
	    g++ gfortran file binutils \
	    openblas-dev \
	    python3-dev=3.8.2-r0 \
	    gcc \
	    build-base \
	    libpng-dev \
	    musl-dev \
	    freetype-dev \
	    libxml2-dev \
	    libxslt-dev \
	    libgcc libquadmath \
	    libgfortran \
	    lapack-dev \
            linux-headers \
	&& ln -s /usr/include/locale.h /usr/include/xlocale.h \
	&& pip3 install numpy==1.18.3 \
	&& pip3 install scipy==1.3.3 \
	&& pip3 install pandas==1.0.3 \
	&& pip3 install kiwisolver==1.1.0 \
	&& pip3 install matplotlib==3.2.1 \
	&& pip3 install mne \
        && pip3 install bids_validator \
        && pip3 install psutil==5.7.0 \
	&& /pymef/setup.py install \
	&& rm -rf /pymef* \
	&& rm -r /root/.cache \
	&& find /usr/lib/python3.*/ -name 'tests' -exec rm -r '{}' + \
	&& find /usr/lib/python3.*/site-packages/ -name '*.so' -print -exec sh -c 'file "{}" | grep -q "not stripped" && strip -s "{}"' \; \
	&& rm /usr/include/xlocale.h \
	&& apk del .build-deps

RUN apk add --update npm
#	&& npm install -g bids-validator

# 
ENV PYTHONPATH=""

#
RUN mkdir -p /scripts
COPY n1detect_*.py /scripts/
RUN chmod +x /scripts/n1detect_run.py

#
COPY version /scripts/version
COPY ./functions /scripts/functions

# 
ENTRYPOINT ["/scripts/n1detect_run.py"]
