FROM nvcr.io/nvidia/tensorrt:22.04-py3

# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:graphics-drivers
RUN apt-get update && apt-get install dialog apt-utils debconf-utils -y
#ARG USERNAME
ARG USER=light
ARG GROUP=developer
ARG UID=1000
ARG GID=1000
RUN apt-get update && \
      apt-get -y install sudo
# RUN apt-get update && apt-get install dialog apt-utils -y
# RUN apt-get update && apt-get install -y debconf-utils
#add user 
RUN useradd -m light && echo "light:light" | chpasswd && adduser light sudo && su light
# # persist bash history for root user
# #https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo $SNIPPET >> "/root/.bashrc"
#RUN apt-get update && apt-get install -y python3-rosdep && rm /etc/ros/rosdep/sources.list.d/20-default.list 
RUN apt-get install -y libeigen3-dev && apt-get install -y libqt5widgets5 libtinyxml-dev
ENV EIGEN3_INCLUDE_DIR /usr/include/eigen3
ENV PIP_ROOT_USER_ACTION=ignore
# install cmake
# ARG CMAKE_VERSION=3.23
# ARG CMAKE_BUILD=2
# RUN apt-get update && apt-get install -y libssl-dev && wget https://cmake.org/files/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.$CMAKE_BUILD.tar.gz &&  tar -xzvf cmake-$version.$build.tar.gz \
#     && ./bootstrap && make -j4 && make install 

# RUN cd /opt/tensorrt/python && ./python_setup.sh
#RUN echo "y" | conda install ipywidgets colored onnxruntime 
# use the above commands only with pytorch container

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt update && apt install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-foxy-desktop
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin
RUN /opt/tensorrt/install_opensource.sh 


USER light
WORKDIR /home/light
ENV ROS_PYTHON_VERSION 3
RUN hash -r 


#for jupyter notebook
#COPY ./ros2-deb.sh /home/light
#RUN echo "light" | sudo -S ./ros2-deb.sh


#  RUN echo "light" | sudo -S apt-get -y update &&  mkdir -p /home/light/ros2_foxy/src \
#     && cd /home/light/ros2_foxy && wget https://raw.githubusercontent.com/ros2/ros2/foxy/ros2.repos \
#     && vcs import src < ros2.repos

# #RUN echo "light" | sudo -S apt-get -y update && sudo apt-get -y upgrade 
# RUN echo "light" | sudo -S rosdep init 
# RUN rosdep update 
# RUN echo "light" | sudo -S apt-get -y update && rosdep install --from-paths /home/light/ros2_foxy/src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-5.3.1 urdfdom_headers"

#     #sudo apt remove shiboken2 libshiboken2-dev libshiboken2-py3-5.14
# #pip3 install --user shiboken2

# RUN  echo "light" | sudo -S apt-get -y update && cd /home/light/ros2_foxy/ \
# && colcon build --symlink-install

#COPY ./ros2-foxy-20220208-linux-focal-amd64.tar.bz2 /home/light/ 
#RUN tar -xvf ros2-foxy-20220208-linux-focal-amd64.tar.bz2 && mv ros2-linux ros2_foxy && echo "light" | sudo -S mv ros2_foxy /opt
#RUN  DEBIAN_FRONTEND=noninteractive && echo "light" | sudo -S apt-get -y update && echo "light" | sudo -S apt -y install ros-foxy-desktop

# RUN source /home/light/ros2_foxy/install/setup.bash 

#CMD bin/bash    

RUN mkdir -p /home/light/ros2_ws/src
WORKDIR /home/light/ros2_ws/src
RUN git clone https://github.com/ros/ros_tutorials.git -b foxy-devel
WORKDIR /home/light/ros2_ws

RUN echo "light" | sudo -S apt-get install python3-rosdep -y
RUN echo  "light" | sudo -S rosdep init
RUN rosdep update
RUN rosdep install -i --from-path src --rosdistro foxy -y
RUN echo "light" | sudo -S apt install python3-colcon-common-extensions -y

COPY ros2_entrypoint.sh /home/light

RUN pip install conan && source ~/.profile 

ENTRYPOINT ["/home/light/ros2_entrypoint.sh"]
CMD ["bash"]
