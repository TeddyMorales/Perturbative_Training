import pybullet as p
import time
import pybullet_data
import math
from datetime import datetime

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
p.loadURDF("plane.urdf")
sawyerId = p.loadURDF("rudis_magic_sawyer.urdf", [0,0,0])
p.resetBasePositionAndOrientation(sawyerId, [0,0,0],[0,0,0,1])
# populate the following list with the indices of the 7 moveable joints 
self.joints = [4,9,10,11,12,14,17]
# set joint limits 
self.upper_limits = np.array([-3.0503,-3.8095,-3.0426,-3.0439,-2.9761,-2.9761,-4.7124])
self.lower_limits = np.array([3.0503,2.2736,3.0426,3.0439,2.9761,2.9761,4.7124])

# move to cartesian destination
move(destination)

while 1:
    p.stepSimulation()


def move(self, cartesian_destination):
    # Given a space we want to get to in cartesian space lets find desination joint state 
    target_joint_state = p.calculateInverseKinematics(sawyerId, self.joints[-1], cartesian_destination, lowerLimits=self.lower_limits, upperLimits = self.upper_limits) 

    mode = p.POSITION_CONTROL # Control Mode, position because we have joint states not velocities 
    velocity = 10
    force = 0 # Max force 
    for i in enumerate(self.joints): # loop through each joint and decide where it needs to end up 
        p.SetJointMotorControl2(sawyerId, self.joints[i], target_joint_state[self.joints[i]], velocity, mode, force) 

    return
