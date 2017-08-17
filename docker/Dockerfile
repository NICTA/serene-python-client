#
# Docker file for the serene python client in combination with cp solver
#
FROM ubuntu

MAINTAINER Data61 CSIRO

# Python version
ENV PYTHON_VERSION 3.5.3
ENV PYTHON_PIP_VERSION 9.0.1
# configure python encodings
ENV PYTHONIOENCODING=utf-8
ENV PYTHONENCODING=utf-8

# Pick up some dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3.5 \
        python3.5-dev \
        rsync \
        apt-transport-https \
        software-properties-common \
        unzip \
        nano \
        flex && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin \
#	&& { [ -e easy_install ] || ln -s easy_install-* easy_install; } \
	&& ln -s idle3 idle \
	&& ln -s pydoc3 pydoc \
	&& ln -s python3 python \
	&& ln -s python3-config python-config

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

#
# Install python dependencies
#
RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        Pillow \
        pandas \
        requests \
        rdflib \
        networkx \
        pyyaml && \
    python3 -m ipykernel.kernelspec

#
# Install pygraphviz
#
RUN apt-get update && apt-get install -y graphviz libgraphviz-dev
#RUN apt-get install libgraphviz-dev
#RUN apt-get install libgraphviz4
RUN pip3 install pygraphviz

#
# Next we create a serene user...
#
ENV username serene
RUN useradd -ms /bin/bash $username && echo "$username:$username" | chpasswd && adduser $username sudo
RUN mkdir -p /home/$username && chown -R $username:$username /home/$username
WORKDIR /home/$username

#
# Install boost
#
COPY boost_1_64_0.tar.gz .
RUN tar vxzf /home/$username/boost_1_64_0.tar.gz -C /usr/local
ENV BOOST_ROOT /usr/local/boost_1_64_0
RUN cd $BOOST_ROOT/ && \
 ./bootstrap.sh && \
 ./b2 install

# Set the locale
RUN apt-get update && apt-get install -y --no-install-recommends locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

#
# clean up...
#
RUN apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

#
# Copy and configure stp
RUN mkdir -p /home/$username/stp && chown -R $username:$username /home/$username/stp
COPY stp/ /home/$username/stp/
RUN cd /home/$username/stp/minizinc && \
    mkdir -p chuffed-rel2onto_20170718 && chown -R $username:$username chuffed-rel2onto_20170718 && \
    mkdir -p chuffed-rel2onto_20170722 && chown -R $username:$username chuffed-rel2onto_20170722 && \
    mkdir -p chuffed-rel2onto_20170728 && chown -R $username:$username chuffed-rel2onto_20170728 && \
    unzip chuffed-rel2onto_20170718.zip -d ./chuffed-rel2onto_20170718 && \
    unzip chuffed-rel2onto_20170722.zip -d ./chuffed-rel2onto_20170722 && \
    unzip chuffed-rel2onto_20170728.zip -d ./chuffed-rel2onto_20170728


RUN cd /home/$username/stp/minizinc/chuffed-rel2onto_20170718 && make
RUN cd /home/$username/stp/minizinc/chuffed-rel2onto_20170722 && make
RUN cd /home/$username/stp/minizinc/chuffed-rel2onto_20170728 && make

#
# Copy and install MiniZinc
#
COPY ./MiniZincIDE-2.1.4-bundle-linux-x86_64.tgz /home/$username/
RUN tar vxzf MiniZincIDE-2.1.4-bundle-linux-x86_64.tgz
ENV PATH $PATH:/usr/local/boost_1_64_0:/home/$username/MiniZincIDE-2.1.4-bundle-linux-x86_64
#ENV PATH $PATH:/home/$username/MiniZincIDE-2.1.4-bundle-linux-x86_64


#
# Copy and install serene-python-client
#
USER $username
RUN mkdir -p serene-python-client && chown -R $username:$username serene-python-client
COPY serene-python-client/ serene-python-client/
# Install them
USER root
RUN pip3 install serene-python-client/*.gz && rm -r serene-python-client/


RUN chown -R $username:$username /home/$username/stp/* && \
    chown -R $username:$username /home/$username/stp/resources/*
USER $username
RUN mkdir -p /home/$username/stp/resources/results && chown -R $username:$username /home/$username/stp/resources/results
ENV SERENE_PYTHON /home/$username/
ENV SERENE_BENCH /home/$username/stp/resources

CMD ["/bin/bash"]

#docker run -it -u root --net=host -v /home/ubuntu/serene-stp/results:/home/serene/stp/resources/results --name benchmark-instance serene-stp