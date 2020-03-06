FROM alpine:3.11

RUN apk add --no-cache libpng freetype libstdc++ openblas libxml2 libxslt && \
	apk add --no-cache --virtual .build-deps \
	    g++ gfortran file binutils \
	    openblas-dev \
	    python3-dev \
	    gcc \
	    build-base \
	    libpng-dev \
	    musl-dev \
	    freetype-dev \
	    libxml2-dev \
	    libxslt-dev && \
	ln -s /usr/include/locale.h /usr/include/xlocale.h \
	&& pip3 install numpy \
	&& pip3 install scipy==1.3.3 \
	&& pip3 install pandas \
	&& pip3 install matplotlib \
	&& pip3 install joblib \
	&& rm -r /root/.cache \
	&& find /usr/lib/python3.*/ -name 'tests' -exec rm -r '{}' + \
	&& find /usr/lib/python3.*/site-packages/ -name '*.so' -print -exec sh -c 'file "{}" | grep -q "not stripped" && strip -s "{}"' \; \
	&& rm /usr/include/xlocale.h \
	&& apk del .build-deps
    
ENV PYTHONPATH=""

COPY run.py /run.py

COPY version /version

ENTRYPOINT ["/run.py"]
