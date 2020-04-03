import pybullet as p
import random
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", [0, 0, -.98])


# load robot
# load robot
bot_start_position = [0, 0, 0]
bot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
# TODO define robot position and orientation
bot = p.loadURDF("/u/teddy/Robotics_Research/catkin_ws/Perturbative_Training/rudis_magic_sawyer.urdf",
                 bot_start_position, bot_start_orientation)


sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])
colTableId = p.createCollisionShape(p.GEOM_BOX,
                                    halfExtents=[0.8, 0.2, 0.38])

mass = 1
visualShapeId = -1

link_Masses = [1]
linkCollisionShapeIndices = [colBoxId]
linkVisualShapeIndices = [-1]
linkPositions = [[0, 0, 0.11]]
linkOrientations = [[0, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[0, 0, 1]]


sphereBasePosition = [
    0, 2, 1
]

sphere2BasePosition = [
    -2, 2, 1
]

boxBasePosition = [
    2, 2, 1
]

tableBasePosition = [
    0.5, 0, 1
]

baseOrientation = [0, 0, 0, 1]

sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, sphereBasePosition,
                              baseOrientation)
sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, sphere2BasePosition,
                              baseOrientation)

sphereUid = p.createMultiBody(mass,
                              colBoxId,
                              visualShapeId,
                              boxBasePosition,
                              baseOrientation,
                              linkMasses=link_Masses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=indices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis)

# table
sphereUid = p.createMultiBody(500,
                              colTableId,
                              visualShapeId,
                              tableBasePosition,
                              baseOrientation,
                              linkMasses=link_Masses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=indices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis)

p.changeDynamics(sphereUid,
                 -1,
                 spinningFriction=0.001,
                 rollingFriction=0.001,
                 linearDamping=0.0)
for joint in range(p.getNumJoints(sphereUid)):
    p.setJointMotorControl2(
        sphereUid, joint, p.VELOCITY_CONTROL, targetVelocity=1, force=10)

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

p.getNumJoints(sphereUid)
for i in range(p.getNumJoints(sphereUid)):
    p.getJointInfo(sphereUid, i)

while (1):
    keys = p.getKeyboardEvents()
    print(keys)

    time.sleep(0.01)
