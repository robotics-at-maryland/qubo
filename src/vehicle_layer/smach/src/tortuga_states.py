#! /usr/bin/env python

# State machine for tortuga 2016 competition
# States given by kanga can be found at: imgur.com/C5KLSCJ

import roslib; roslib.load_manifest('smach')
import rospy
import smach
import smach_ros

#TODO
class gate_search(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes = ['found', 'not_found'])
        
        def execute(self):

class gate_travel(smach.State):
    def __init__(self):
        (self,
        outcomes = ['passed', 'failed'])

class sonar_search(smach.State):
    def __init__(self):
        (self,
         outcomes = ['found', 'not_found'],
         output_keys = ['sonar_position_vector'])
        
class sonar_travel(smach.State):
    def __init__(self):
        (self,
         outcomes = ['centered', 'not_centered'],
         input_keys = ['sonar_position_vector'])
        
class search_bin_cover(smach.State):
    def __init__(self):
        (self,
         outcomes = ['bins', 'octagon'])
        
class do_bin_task(smach.State):
    def __init__(self):
        (self,
         outcomes = ['placed', 'lifted', 'incomplete'])

class surface(smach.State):
    def __init__(self):
        (self,
         outcomes = ['inside', 'oh_no'])

def main():
    rospy.init_node('toruga_ai_node')

    sm = smach.StateMAchine(outcomes=['done'])
    sm.userdata.sonar_position_vector = [0,0,0]

    with sm:
        smach.StateMachine.add('gate_search', gate_search(),
                               transitions={'found':'gate_travel',
                                            'not_found':'gate_search'})
        smach.StateMachine.add('gate_travel', gate_travel(),
                               transitions={'passed':'sonar_search',
                                            'failed':'gate_search'})
        smach.StateMachine.add('sonar_search', sonar_search(),
                               transitions={'found':'sonar_travel',
                                            'not_found':'sonar_search'},
                               remapping={'sonar_position_vector':'sonar_position_vector'})
        smach.StateMachine.add('sonar_travel', sonar_travel(),
                               transitions={'centered':'search_bin_cover',
                                            'not_centered':'sonar_travel'},
                               remapping={'sonar_position_vector':'sonar_position_vector'})
        smach.StateMachine.add('search_bin_cover', search_bin_cover(),
                               tranistions={'bins':'do_bin_task',
                                            'octagon':'surface',})
        smach.StateMachine.add('do_bin_task', do_bin_task(),
                               transitions={'placed':'sonar_search',
                                            'lifted':'sonar_search',
                                            'incomplete':'sonar_search'})
        smach.StateMachine.add('surface', surface(),
                               transitions={'inside':'sonar_search',
                                            'oh_no':'done'})

