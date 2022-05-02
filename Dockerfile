FROM python:3.10-slim-bullseye
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
	&& pip3 install numpy==1.22.3 \
	&& pip3 install scipy==1.8.0 \
	&& pip3 install pandas==1.4.2 \
	&& pip3 install matplotlib==3.5.1 \
	&& pip3 install mne==1.0.2 \
        && pip3 install bids_validator==1.9.3 \
        && pip3 install psutil==5.9.0 \
	&& pip3 install pymef \
	&& rm -r /root/.cache

# 
ENV PYTHONPATH=""

#
RUN mkdir -p /app
COPY main_command.py /app/main_command.py
COPY version /app/version
COPY ./erdetect /app/erdetect

#
RUN chmod +x /app/main_command.py

# 
ENTRYPOINT ["/app/main_command.py"]
