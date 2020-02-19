# Base off of Dockerfile for ros:perception
FROM ros:kinetic-robot-xenial

# Set docker to use bash for all RUN commands
# Needed because we have to source a lot of stuff
SHELL ["/bin/bash", "-c"]
WORKDIR /src/qubo

# A script to setup the environment
COPY ./ram_entrypoint.sh /
ENTRYPOINT ["/ram_entrypoint.sh"]
RUN echo 'source /ram_entrypoint.sh' > ~/.bashrc

# Directories needed later
RUN mkdir -p /src/vimba /catkin_ws/src

# Standard dependencies
RUN apt-get update && apt-get install -y \
	autoconf \
	bison \
	build-essential \
	curl \
	doxygen \
	flex \
	libelf-dev \
	libncurses5-dev \
	libopencv-dev \
	libusb-1.0-0-dev \
	protobuf-compiler \
	protobuf-c-compiler \
	sudo \ 
	texinfo

# ROS packages
RUN  apt-get install -y --allow-unauthenticated \
	python-wstool \
	python-catkin-tools \
	python-catkin-lint \
	ros-kinetic-desktop-full \
	ros-kinetic-underwater-vehicle-dynamics  \
	ros-kinetic-robot-localization \
	ros-kinetic-gazebo-ros \
	ros-kinetic-gazebo-plugins \
	ros-kinetic-gazebo-msgs \
	ros-kinetic-gazebo-plugins \
	ros-kinetic-gazebo-ros \
	ros-kinetic-gazebo-ros-control \
	ros-kinetic-gazebo-ros-pkgs \
	ros-kinetic-effort-controllers \
	ros-kinetic-image-pipeline \
	ros-kinetic-image-common \
	ros-kinetic-perception \
	ros-kinetic-perception-pcl \
	ros-kinetic-robot-state-publisher \
	ros-kinetic-ros-base \
	ros-kinetic-viz \
	ros-kinetic-hector-localization \
	ros-kinetic-joy \
	ros-kinetic-joy-teleop \
	ros-kinetic-video-stream-opencv

# Other packages
RUN apt-get install -y \
	gazebo7

RUN source /opt/ros/kinetic/setup.bash && \
	cd /catkin_ws/src && \
	catkin_init_workspace && \
	wstool init && \
	git clone https://github.com/uuvsimulator/uuv_simulator && \
	cd uuv_simulator && \
	git checkout 9078b8890efb9ad4aa18bb1407e5605883d0d272 && \
	cd /catkin_ws && \
	catkin_make && catkin_make install

RUN source /opt/ros/kinetic/setup.bash && cd /src && \
	git clone https://github.com/robotics-at-maryland/qubo && \
	rosdep install -y -r --reinstall --from-paths qubo/src

ADD vimba.tgz /src/vimba/

RUN cd /src/vimba && \
	cd /src/vimba/Vimba_2_1/VimbaCPP && \
	cp -r ./DynamicLib/x86_64bit/* /usr/local/lib && \
	mkdir -p /usr/local/include/VimbaCPP/Include && \
	cp -r Include /usr/local/include/VimbaCPP/ && \
	cd /src/vimba/Vimba_2_1/VimbaImageTransform/ && \
	cp -r ./DynamicLib/x86_64bit/* /usr/local/lib && \
	mkdir -p /usr/local/include/VimbaImageTransform && \
	cp -r Include /usr/local/include/VimbaImageTransform/ && \
	mkdir -p /usr/local/include/VimbaC && \
	cd /src/vimba/Vimba_2_1/VimbaC && \
	cp -r Include /usr/local/include/VimbaC/


CMD ["catkin_make"]
