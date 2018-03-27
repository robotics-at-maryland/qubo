#! /usr/bin/env python
import rospy
import cmd
from std_msgs.msg import Float64
import sys

qubo_namespace = '/qubo/'


class EmbeddedController(cmd.Cmd):

    def __init__(self):
        cmd.Cmd.__init__(self).__init__()
        rospy.init_node('embedded_controller')
        self.r = rospy.Rate(10)

        # publishers and things
        self.yaw   = rospy.Publisher(qubo_namespace + 'yaw_cmd', Float64,
                                     queue_size=5)
        self.pitch = rospy.Publisher(qubo_namespace + 'pitch_cmd', Float64,
                                     queue_size=5)
        self.roll  = rospy.Publisher(qubo_namespace + 'roll_cmd', Float64,
                                     queue_size=5)
        self.depth = rospy.Publisher(qubo_namespace + 'depth_cmd', Float64,
                                     queue_size=5)
        self.surge = rospy.Publisher(qubo_namespace + 'surge_cmd', Float64,
                                     queue_size=5)
        self.sway  = rospy.Publisher(qubo_namespace + 'sway_cmd', Float64,
                                     queue_size=5)

    def do_yaw(self, thrust):
        """publish the given [thrust] to the yaw topic"""
        self.yaw.publish(thrust)

    def do_pitch(self, thrust):
        """publish the given [thrust] to the pitch topic"""
        self.pitch.publish(thrust)

    def do_roll(self, thrust):
        """publish the given [thrust] to the roll topic"""
        self.roll.publish(thrust)

    def do_depth(self, thrust):
        """publish the given [thrust] to the depth topic"""
        self.depth.publish(thrust)

    def do_surge(self, thrust):
        """publish the given [thrust] to the surge topic"""
        self.surge.publish(thrust)

    def do_sway(self, thrust):
        """publish the given [thrust] to the sway topic"""
        self.sway.publish(thrust)

    def do_exit(self, *args):
        sys.exit()

    def do_quit(self, *args):
        sys.exit()

if __name__ == "__main__":
    # run the interpreter
    EmbeddedController().cmdloop()
