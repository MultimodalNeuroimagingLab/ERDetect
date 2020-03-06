FROM debian:buster-slim

# Install python 3 and numpy
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-numpy && \
    apt-get remove -y python3-pip && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PYTHONPATH=""

COPY run.py /run.py

COPY version /version

ENTRYPOINT ["/run.py"]
