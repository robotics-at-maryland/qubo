# Check if the script is run as root
if [[ $EUID -ne 0 ]] ; then
	echo "Please run this script as root"
	exit 1
fi

# Install QtCreator and its ROS plugin.
# Due to a quirk of the plugin, qtcreator MUST be installed twice.
read -p "Do you want to install optional IDE QtCreator and associated ROS plugin? [Y/n]" -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
	sudo apt install qtcreator;
	# This command broke the install on ubuntu 16.04.2
	# sudo add-apt-repository ppa:beineri/opt-qt57-xenial;
	# https://github.com/ros-industrial/ros_qtc_plugin/wiki/1.-How-to-Install-(Users)
	# sudo add-apt-repository --remove ppa:beineri/opt-qt57-xenial
	sudo add-apt-repository ppa:levi-armstrong/qt-libraries-xenial
	sudo add-apt-repository ppa:levi-armstrong/ppa
	sudo apt-get update && sudo apt-get install qt57creator-plugin-ros
	sudo apt install qtcreator
fi
