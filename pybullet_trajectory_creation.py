import pybullet as p
import time
import pybullet_data
import math
from datetime import datetime
import numpy as np



def move(cartesian_destination, upper_limits, lower_limits, sawyerId, joints):
    # Given a space we want to get to in cartesian space lets find desination joint state 
    target_joint_state = p.calculateInverseKinematics(sawyerId, 18, cartesian_destination, lowerLimits=lower_limits, upperLimits = upper_limits) 

    #print(target_joint_state)


    mode = p.POSITION_CONTROL # Control Mode, position because we have joint states not velocities 
    velocity = 10
    force = 5 # Max force 
    for i in range(len(joints)): # loop through each joint and decide where it needs to end up 
        p.setJointMotorControl2(sawyerId, joints[i], p.POSITION_CONTROL, target_joint_state[i]) 

    return

def loadObjects():
	sphereRadius = 0.05
	colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
	colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])
	colTableId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[0.8, 0.2, 0.45])

	mass = 1
	visualShapeId = -1

	sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, [.5, 0, -.05])                                      
	sphereUid = p.createMultiBody(mass, colBoxId, visualShapeId, [.5, 1, -.05])
	sphereUid = p.createMultiBody(500, colTableId, visualShapeId, [.5, .5, -.5], [1,1,0,0])

	p.changeDynamics(sphereUid, -1, spinningFriction=0.001, rollingFriction=0.001, linearDamping=0.0)




physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf", [0,0,-.98])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
sawyerId = p.loadURDF("rudis_magic_sawyer.urdf", [0,0,0])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

p.resetBasePositionAndOrientation(sawyerId, [0,0,0],[0,0,0,1])

# populate the following list with the indices of the 7 moveable joints 
joints = [4,5,9,10,11,12,14,17,23,25]
# set joint limits 
upper_limits = np.array([-3.0503,-3.8095,-3.0426,-3.0439,-2.9761,-2.9761,-4.7124])
lower_limits = np.array([3.0503,2.2736,3.0426,3.0439,2.9761,2.9761,4.7124])

#load Ball, table and goal
loadObjects();

p.setGravity(0,0,-1)
t = 0.
# prevPose = [0,0,0]
# prevPose1 = [0,0,0]
# hasPrevPose = 0

useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 10


# move to cartesian destination
i = .01


while 1:
	if (useRealTimeSimulation):
		dt = datetime.now()
		t = (dt.second/60.)*2.*math.pi
		print (t)
	else:
		t = t + 0.01
		time.sleep(0.01)
	


	destination = [i,i,i]
	move(destination, upper_limits, lower_limits, sawyerId, joints)
	i += .00001
	print(i)

	# ls = p.getLinkState(sawyerId,sawyerEndEffectorIndex)
	# if (hasPrevPose):
	# 	p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
	# 	p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
	# prevPose=pos
	# prevPose1=ls[4]
	# hasPrevPose = 1
