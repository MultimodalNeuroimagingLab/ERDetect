FROM python:3.9-slim-bullseye
LABEL MAINTAINER="Max van den Boom <m.a.vandenboom84@gmail.com>"

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# ensure UTF-8 is working with slim images and minimum locales, taken from the "official" python Docker images
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# This hack is widely applied to avoid python printing issues in docker containers.
# See: https://github.com/Docker-Hub-frolvlad/docker-alpine-python3/pull/13
ENV PYTHONUNBUFFERED=1

# build and install numpy, scipy, mne, bids_validator and pymef
RUN pip3 install --upgrade pip \
	&& pip3 install numpy==1.21.1 \
	&& pip3 install scipy==1.7.1 \
	&& pip3 install pandas==1.3.0 \
	&& pip3 install kiwisolver==1.3.2 \
	&& pip3 install matplotlib==3.4.2 \
	&& pip3 install mne==0.23.4 \
        && pip3 install bids_validator==1.8.4 \
        && pip3 install psutil==5.8.0 \
	&& pip3 install pymef \
	&& rm -r /root/.cache

#RUN apk add --update npm
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
