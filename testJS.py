import pybullet as p
import time
import pybullet_data
import math
from datetime import datetime
import numpy as np

def clip_joint_velocities(velocities):
    # Clips joint velocities into a valid range.
    for i in range(len(velocities)):
        if velocities[i] >= 1.0:
            velocities[i] = 1.0
        elif velocities[i] <= -1.0:
            velocities[i] = -1.0
    return velocities


def get_control(target_joint_state, curJointPos, rotation=None):
    # get target velocities for motion
    velocities = np.zeros(10)
    deltas = curJointPos - target_joint_state
    for i, delta in enumerate(deltas):
        velocities[i] = -2. * delta  # -2. * delta
    velocities = clip_joint_velocities(velocities)
    return velocities

def get_pos(obj_uid):
    return p.getBasePositionAndOrientation(obj_uid)    

# input is trajectory, numpy array of postitions at each time step
# while time is going, loop through the positions
# and then extract what happens
def move(joint_poses, upper_limits, lower_limits, sawyerId, joints):
    jd = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    # Given a space we want to get to in cartesian space lets find desination joint state 
    curJointPos = []
    for joint in joints:
        curJointPos.append(p.getJointState(sawyerId, joint)[0])
    curJointPos = np.array(curJointPos)

    step = .01

    #should prob be way lower, 0.01
    error = .1

    
    delta = curJointPos - joint_poses
    #print ("i'm in the move method")


    # TODO: account for hitting table and time being up
    while(np.linalg.norm(delta) > error):
        #print("ball pos = " + get_ball_pos())
        # print ("i'm in the while loop")
        curJointPos += step * joint_poses
        # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(p.getNumJoints(sawyerId)):
            #  print ("I'm in the for loop")
            jointInfo = p.getJointInfo(sawyerId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(sawyerId, i, curJointPos[qIndex-7])
        delta = delta = curJointPos - joint_poses
    return


def loadObjects():
    # returns golf ball uid so we can use it when requesting position
    sphereRadius = 1000000000.05
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])
    colClubId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.03, .03, .03])
    colTableId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.2, 0.45])

    mass = 1
    visualShapeId = -1

    golf_ball_uid = p.createMultiBody(mass, colSphereId, visualShapeId, [.5, 0, -.05])
    targetBox1 = p.createMultiBody(mass, colBoxId, visualShapeId, [.5, 1, -.05])
    targetBox2 = p.createMultiBody(mass, colClubId, visualShapeId, [.5, -.2, -.05])
    table = p.createMultiBody(500, colTableId, visualShapeId, [.5, .5, -.5], [1, 1, 0, 0])

    p.changeDynamics(golf_ball_uid, -1, spinningFriction=0.001, rollingFriction=0.001, linearDamping=0.0)
    return golf_ball_uid


# DO NOT DELETE: global variables: positions of the goal
goalX = .5
goalY = 1
goalZ = -.05

# function that returns distance between ball and goal 
def get_distance():
    ball_pos = np.array(get_pos(golf_ball_uid)[0])
    goal_pos = np.array([goalX, goalY, goalZ])
    print(np.linalg.norm(ball_pos-goal_pos))
    return np.linalg.norm(ball_pos-goal_pos)
    
# checks if ball has hit target position    
def is_success():
    return get_distance() == 0

def setEnvironment():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load plane and sawyer
    p.loadURDF("plane.urdf", [0, 0, -.98])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    sawyerId = p.loadURDF("C:/Users/Jain/aditij/catkin-ws/src/rudis_magic_sawyer.urdf", [0, 0, 0], useFixedBase = 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.resetBasePositionAndOrientation(sawyerId, [0, 0, 0], [0, 0, 0, 1])

    # set up joints
    joints = [4, 5, 9, 10, 11, 12, 14, 17, 23, 25]

    upper_limits = np.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
    lower_limits = np.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])

    # loadball and bocks
    golf_ball_uid = loadObjects()

    p.setGravity(0,0,-10)


    # Make sure simulator is going at real-time speed
    useRealTimeSimulation = 1
    p.setRealTimeSimulation(useRealTimeSimulation)

    starting_joint_angles = [-0.041662954890248294, 0, -1.0258291091425074, 0.0293680414401436, 2.17518162913313, -0.06703022873354225, 0.3968371433926965, 1.7659649178699421, 0, 0]
    p.setJointMotorControlArray(sawyerId, joints, p.POSITION_CONTROL, starting_joint_angles)

    miniTraj = [[.5, 0, .3], [.5, 0, .2], [.5, 0, .1]]

    p.setJointMotorControl2(sawyerId, 23, p.POSITION_CONTROL, targetPosition=1) 
    p.setJointMotorControl2(sawyerId, 25, p.POSITION_CONTROL, targetPosition=-1) 

    destination = [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
    destination2 = [-.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5]
    return golf_ball_uid

def simulate():
    # Create a variable to track time
    t = 0
    # define max simluation time
    max_time = 500
    # continue if no success, there's still time, still on the table
    # TODO: find sphere initial height and ball initial height
    while not is_success() and t < max_time and get_pos(golf_ball_uid)[0][1] == 0.5:
        # move(np.array(destination), upper_limits, lower_limits, sawyerId, joints)
        # move(np.array(destination2), upper_limits, lower_limits, sawyerId, joints)
        print (get_pos(golf_ball_uid)[0])
        #exits when there's a success, returns 1
        if is_success():
            return is_sucess()
            
# load simulated enviroment
golf_ball_uid = setEnvironment()        
# run simulation
simulate()

# prints final ball position
print(get_pos(golf_ball_uid))

print("DONE MOTION")
