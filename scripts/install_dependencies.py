import os.path
import platform
import subprocess

APT_GET_LIST = ['ros-indigo-desktop',
                'doxygen',
                'libarmadillo-dev',
                'python-pip']

PIP_LIST = []

def ros_preinstall():
    linux_distro = platform.linux_distribution()
    #The string that will be added to the apt-get sources list
    ros_deb_url = 'deb http://packages.ros.org/ros/ubuntu %s main' % linux_distro[2]
    #The file path of the ROS apt-get sources list
    ros_src_list = '/etc/apt/sources.list.d/ros-latest.list'
    #The URL for the ROS apt-get key
    ros_key_url = 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key'

    #If the ROS apt-get sources list has already been created, we can skip this step.
    if os.path.isfile(ros_src_list):
        return

    print('Adding ROS to apt-get sources list...')
    subprocess.call("sudo sh -c 'echo \"%s\" > %s'" % (ros_deb_url, ros_src_list), shell=True)
    subprocess.call('wget %s -O - | sudo apt-key add -' % ros_key_url, shell=True)
    subprocess.call('sudo apt-get update'.split())

def ros_postinstall():
    #Rosdep will create this directory when it is initialized.
    #If it's there, we can skip this step.
    if os.path.isdir('/etc/ros/rosdep/'):
        return

    print('Initializing Rosdep...')
    subprocess.call('sudo rosdep init'.split())
    subprocess.call('rosdep update'.split())

def main():
    ros_preinstall()
    subprocess.call('sudo apt-get install'.split() + APT_GET_LIST)
    #subprocess.call('sudo pip install'.split() + PIP_LIST)
    ros_postinstall()

if __name__ == '__main__':
    main()
