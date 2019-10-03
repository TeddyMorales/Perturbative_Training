import pybullet as p 
import time 
import pybullet_data 
import random

class huge_robot(object):


    def __init__(self):
        physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optional 

        pass

    def pertubation(self, n, m, enviro_vars):
        '''performs pertubative training by changing variables randomly 
            within a range for each variables parameters  '''
        for (i in range n):
            
                
