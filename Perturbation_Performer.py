import pybullet
import random
import time
import pybullet_data

class golfbot(object):

    def __init__(self):
        # set up physics client and environment 
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")

        #load table 
        self.table_position = [0,0,0]
        self.table_orientation = p.getQuaternionFromEuler([0,0,0])
        # TODO find urdf for table and define position
        #self.table = p.loadURDF("table.urdf", self.table_position, self.table_orientaion)
        
        # load robot 
        self.bot_start_position = [0,0,1]
        self.bot_start_orientation = p.getQuaternionFromEuler([0,0,0])
        # TODO define robot position and orientation 
        self.bot = p.loadURDF("rudis_magic_sawyer.urdf", self.bot_start_position, self.bot_start_orientation)

        # load club
        self.club_start_position = [0,0,0]
        self.club_start_orientation = p.getQuaternionFromEuler([0,0,0])
        # TODO define ball position and orientation and load rectangle to serve as club 
        self.club = 
       
        # load goal 
        self.goal_position = [0,0,0]
        self.goal_orientation = p.getQuaternionFromEuler([0,0,0])
        # TODO get urdf for goal (Sphere) and define position/orientation 
        self.goal = 


    def static_perturbation(self, n, m, enviro_vars):
        '''
                Performs static perturbative training by changing variables randomly 
            within a range for each variables parameters and then leaving them static
            for the entirity of the trial. Generates a dict containing identifiable
            environment variables as the keys and a probabilistic movement primitive as
            the value.

         '''
        
        # loop through each variable and change them randomly 
        for i in range n:
            # simulate m times for each configuration 
            for j in range m:
                # simulate a trajectory, activity 
                for i in range (10000):
                    pybullet.stepSimulation()
                    time.sleep(1./240.)
            # for each simulation, record a trajectory and return trajectories to pro-mp creator 
            
            #for each trajectory use reinforcement learning to determine whether it was a success(made a basket) or failure (missed) 


    def dynamic_perturbation(self, n, m, enviro_vars, rate_of_change):
        '''
                    Performs dynamic perturbative training by changing variables randomly
                    within a range for each variables parameters between iterations of the
                    simulation at the rate given by rate_of_change. Generates a dict 
                    containing identifiable average environment variables as the keys 
                    and a probabilistic movement primitive as the value. 

        '''               
        pass

    
