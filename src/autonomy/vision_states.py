#!/usr/bin/env python
# David Linko
#state server using the clients and action from vision 
#currently have a wait state that goes to a service state or an acton state   

import roslib; roslib.load_manifest('vision')
import rospy
import smach
import smach_ros

from smach_ros import SimpleActionState
from smach_ros import ServiceState

from vision.msg import VisionExampleAction 
from vision.srv import bool_bool

#wait state that the service and action states always go back to 
class Wait(smach.State):
  def __init__(self):
    smach.State.__init__(self, outcomes=['service', 'action'])
    self.counter = 0

  def execute(self, userdata):
      if self.counter%2 == 0:  
        self.counter += 1
        return  'action'
      else:
        self.counter += 1
        return 'service'  

def main():
    rospy.init_node('smach_example_state_machine')

    # Create a SMACH state machine
    sm = smach.StateMachine(['succeeded','aborted','preempted'])

    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('WAIT', Wait(), 
                              transitions={'service':'SERVICE',
                                           'action':'ACTION'})

        smach.StateMachine.add('SERVICE',
                           ServiceState('buoy_detect',
                                        bool_bool),
                           transitions={'succeeded':'WAIT',
                                        'aborted':'WAIT',
                                        'preempted':'WAIT'})

        smach.StateMachine.add('ACTION',
                           SimpleActionState('vision_example',
                                             VisionExampleAction),
                           transitions={'succeeded':'WAIT',
                                        'aborted':'WAIT',
                                        'preempted':'WAIT'})

    # Execute SMACH plan
    outcome = sm.execute()

if __name__ == '__main__':
    main()

