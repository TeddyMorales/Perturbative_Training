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


def move(cartesian_destination, upper_limits, lower_limits, sawyerId, joints):
    jd = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
          0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    # Given a space we want to get to in cartesian space lets find desination joint state
    curJointPos = []
    for joint in joints:
        curJointPos.append(p.getJointState(sawyerId, joint)[0])
    curJointPos = np.array(curJointPos)

    step = .01
    error = .01

    jointPoses = np.array(p.calculateInverseKinematics(
        sawyerId, 18, cartesian_destination, jointDamping=jd))
    delta = curJointPos - jointPoses

    while(np.linalg.norm(delta) > error):
        curJointPos += step * jointPoses
        # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(p.getNumJoints(sawyerId)):
            jointInfo = p.getJointInfo(sawyerId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(sawyerId, i, curJointPos[qIndex-7])
        delta = delta = curJointPos - jointPoses
    return


def loadObjects():
    sphereRadius = 0.05
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    colBoxId = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])
    colClubId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.03, .03, .03])
    colTableId = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.8, 0.2, 0.45])

    mass = 1
    visualShapeId = -1

    sphereUid = p.createMultiBody(
        mass, colSphereId, visualShapeId, [.5, 0, -.05])
    targetBox = p.createMultiBody(mass, colBoxId, visualShapeId, [.5, 1, -.05])
    targetBox = p.createMultiBody(
        mass, colClubId, visualShapeId, [.5, -.2, -.05])
    table = p.createMultiBody(
        500, colTableId, visualShapeId, [.5, .5, -.5], [1, 1, 0, 0])

    p.changeDynamics(sphereUid, -1, spinningFriction=0.001,
                     rollingFriction=0.001, linearDamping=0.0)


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load plane and sawyer
p.loadURDF("plane.urdf", [0, 0, -.98])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
sawyerId = p.loadURDF("rudis_magic_sawyer.urdf", [0, 0, 0])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

p.resetBasePositionAndOrientation(sawyerId, [0, 0, 0], [0, 0, 0, 1])

# set up joints
joints = [4, 5, 9, 10, 11, 12, 14, 17, 23, 25]

upper_limits = np.array(
    [-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
lower_limits = np.array(
    [3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])

#loadball and bocks
loadObjects()

p.setGravity(0, 0, -10)
t = 0

useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)

starting_joint_angles = [-0.041662954890248294, 0, -1.0258291091425074, 0.0293680414401436,
                         2.17518162913313, -0.06703022873354225, 0.3968371433926965, 1.7659649178699421, 0, 0]
p.setJointMotorControlArray(
    sawyerId, joints, p.POSITION_CONTROL, starting_joint_angles)

miniTraj = [[.5, 0, .3], [.5, 0, .2], [.5, 0, .1]]


# move to cartesian destination
while 1:
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second/60.)*2.*math.pi
        print(t)
    else:
        t = t + 0.01
        time.sleep(0.01)

    # for traj in miniTraj:
    # 	if (useRealTimeSimulation):
    # 		dt = datetime.now()
    # 		t = (dt.second/60.)*2.*math.pi
    # 		print (t)
    # 	else:
    # 		t = t + 0.01
    # 		time.sleep(0.01)
    # 	move(traj, upper_limits, lower_limits, sawyerId, joints)

    p.setJointMotorControl2(sawyerId, 23, p.POSITION_CONTROL, targetPosition=1)
    p.setJointMotorControl2(
        sawyerId, 25, p.POSITION_CONTROL, targetPosition=-1)
    # p.setJointMotorControl2(sawyerId, 23, p.POSITION_CONTROL, force = 0)
    # p.setJointMotorControl2(sawyerId, 25, p.POSITION_CONTROL, force = 0)

    x = float(input())
    y = float(input())
    z = float(input())

    destination = [x, y, z]
    move(np.array(destination), upper_limits, lower_limits, sawyerId, joints)
    print("DONE MOTION")
