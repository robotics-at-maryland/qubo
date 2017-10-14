# /bin/env bash
# Script to start the qubo software system

# to enable this, move it to `/etc/init.d/` and run `update-rc.d /etc/init.d/startup.bash defaults`
echo "Starting Qubo system"
source /home/ubuntu/qubo/devel/setup.bash
nohup roslaunch qubo_launch qubo.launch &
exit 0
