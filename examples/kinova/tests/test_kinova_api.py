#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2
import time
import threading
TIMEOUT_DURATION = 20

def example_api_creation(args):
    '''
    This function creates all required objects and connections to use the arm's services.
    It is easier to use the DeviceConnection utility class to create the router and then 
    create the services you need (as done in the other examples).
    However, if you want to create all objects yourself, this function tells you how to do it.
    '''
    ip = "192.168.8.10"
    PORT = 10000

    # Setup API
    error_callback = lambda kException: print("_________ callback error _________ {}".format(kException))
    transport = TCPTransport()
    router = RouterClient(transport, error_callback)
    transport.connect(ip, PORT)

    # Create session
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = "admin"
    session_info.password = "admin"
    session_info.session_inactivity_timeout = 60000   # (milliseconds)
    session_info.connection_inactivity_timeout = 2000 # (milliseconds)

    print("Creating session for communication")
    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)
    print("Session created")

    # Create required services
    device_config = DeviceConfigClient(router)
    base = BaseClient(router)

    print(device_config.GetDeviceType())
    print("arm state:", base.GetArmState())
    joint_states = base.GetMeasuredJointAngles()
    print(isinstance(joint_states, dict))
    for joint_angle in joint_states.joint_angles:
        print(joint_angle.value)
    
    #ExampleSendGripperCommands(base, 0.995)
    #sendGripperCommand(base)
    #example_angular_action_movement(base, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #example_angular_action_movement(base, [354.17,  357.815,  10.679, 90.142, 358.889, 90.289, 275.476])
    #time.sleep(7.5)

    gripper_request = Base_pb2.GripperRequest()
    gripper_request.mode = Base_pb2.GRIPPER_POSITION
    gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
    print(gripper_measure.finger[0].value)

    # Close API session
    session_manager.CloseSession()

    # Disconnect from the transport object
    transport.disconnect()

def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def example_angular_action_movement(base, target_joint_positions):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = ""
    action.application_data = ""

    actuator_count = base.GetActuatorCount()
    print(actuator_count.count)
    print(len(target_joint_positions))
    assert actuator_count.count == len(target_joint_positions)

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = target_joint_positions[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def ExampleSendGripperCommands(base, target_gripper_position):
    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = target_gripper_position
    base.SendGripperCommand(gripper_command)

    # Close the gripper with position increments
    #print("Performing gripper test in position...")
    #target_position = 0.008
    #gripper_request = Base_pb2.GripperRequest()
    #gripper_request.mode = Base_pb2.GRIPPER_POSITION
    #while True:
    #    gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
    #    if len (gripper_measure.finger):
    #        print("Current position is : {0}".format(gripper_measure.finger[0].value))
    #        if abs(gripper_measure.finger[0].value - target_position) < 0.01:
    #            break
    #        elif gripper_measure.finger[0].value > target_position:
    #            finger.value = max(gripper_measure.finger[0].value - 0.01, target_position)
    #        elif gripper_measure.finger[0].value < target_position:
    #            finger.value = min(gripper_measure.finger[0].value + 0.01, target_position)
    #        base.SendGripperCommand(gripper_command)
    #    else: # Else, no finger present in answer, end loop
    #        break

def sendGripperCommand(base):
    gripper_request = Base_pb2.GripperRequest()
    gripper_request.mode = Base_pb2.GRIPPER_POSITION
    while True:
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
        if len (gripper_measure.finger):
            print("Current position is : {0}".format(gripper_measure.finger[0].value))
            if abs(gripper_measure.finger[0].value - 0.995) < 0.001:
                break
        else: # Else, no finger present in answer, end loop
            break

def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    #import utilities

    # Parse arguments
    args = None
    #args = utilities.parseConnectionArguments()

    # Example core
    example_api_creation(args)
    
if __name__ == "__main__":
    main()