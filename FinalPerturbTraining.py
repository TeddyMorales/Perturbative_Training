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
            
#ProMP Code Starts here

%matplotlib tk
import matplotlib.pyplot as plt
import time
import numpy as np


class AxUpdater(object):
    
    def __init__(self, fig, ax, xlim=None, ylim=None):
        
        self.fig = fig
        self.ax = ax
        
        self.xlim = xlim
        if xlim is not None:
            self.ax.set_xlim(xlim[0],xlim[1])
        
        self.ylim = ylim
        if ylim is not None:
            self.ax.set_ylim(ylim[0],ylim[1])
        
        self.trajs = []
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def new_plot(self):
        self.trajs.append(self.ax.plot([],[], '-o')[0])
        
        
    def update_plot(self,x=None,y=None):
        assert(len(self.trajs))
        if x is not None and y is not None:
            self.trajs[-1].set_data(x, y)
        
        self.fig.canvas.restore_region(self.background)
        # redraw just the points
        self.ax.draw_artist(self.trajs[-1])
        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)
        
    def clean(self):
        for idx in range(len(self.ax.lines)-1,-1,-1):
            if self.ax.lines[idx] not in self.trajs:
                self.ax.lines.pop(idx)
        for idx in range(len(self.ax.collections)-1,-1,-1):
            self.ax.collections.pop(idx)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def clear(self):
        self.ax.set_prop_cycle(None)
        self.trajs=[]
        self.clean()

class DrawOnAx(AxUpdater):

    def __init__(self, fig, ax, xlim=(0.0,1.0),ylim=(0.0,1.0)):
        AxUpdater.__init__(self, fig, ax, xlim, ylim)
        
        self.curr_x = None
        self.curr_y = None
        self.curr_t = None
        self.curr_t_start = None
        self.data = []
        self.cbs = []

        self.do_record = False
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def add_cb(self,cb):
        assert(isinstance(cb,tuple) and len(cb)==3)
        
        self.cbs.append(cb)
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.do_record:
            return
        self.curr_t_start = time.time() 
        self.curr_t = [0.0]
        self.curr_x = [event.xdata]
        self.curr_y = [event.ydata]
        self.new_plot()
        self.update_plot(self.curr_x,self.curr_y)
        self.do_record=True
        
        for cb in self.cbs:
            if cb[0] is not None:
                cb[0](self.curr_t,self.curr_x,self.curr_y)

    def on_move(self, event):
        if event.inaxes != self.ax:
            return
        if not self.do_record:
            return
        
        self.curr_t.append(time.time()-self.curr_t_start)
        self.curr_x.append(event.xdata)
        self.curr_y.append(event.ydata)
        self.update_plot(self.curr_x,self.curr_y)
        
        for cb in self.cbs:
            if cb[1] is not None:
                cb[1](self.curr_t,self.curr_x,self.curr_y)

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        if not self.do_record:
            return
        self.do_record = False
        self.curr_t.append(time.time()-self.curr_t_start)
        self.curr_x.append(event.xdata)
        self.curr_y.append(event.ydata)
        self.update_plot(self.curr_x,self.curr_y)
        self.data.append(np.array([self.curr_t, self.curr_x, self.curr_y]))
        
        for cb in self.cbs:
            if cb[2] is not None:
                cb[2](self.curr_t,self.curr_x,self.curr_y)


    def clear(self):
        self.data = []
        AxUpdater.clear(self)

 /*Motivation
 Given:
 - Trajectory $\traj \in \mathbb{R}^{D\times L}$, with dimension $D$, over $L$ time steps

 Desired: Movement representation that
 - is parametrized
 - is time invariant
 - considers multiple demonstrations
 - captures correlations between DoFs*/

 fig, axs = plt.subplots(3,1,squeeze=False)

 doa = DrawOnAx(fig, axs[0][0])
 uax = AxUpdater(fig, axs[1][0],xlim=(0.0,3.0),ylim=(0.0,1.0))
 uay = AxUpdater(fig, axs[2][0],xlim=(0.0,3.0),ylim=(0.0,1.0))

 doa.add_cb((lambda t,x,y: (uax.new_plot(),uax.update_plot(t,x)), lambda t,x,y: uax.update_plot(t,x), lambda t,x,y: uax.update_plot(t,x)))
 doa.add_cb((lambda t,x,y: (uay.new_plot(),uay.update_plot(t,y)), lambda t,x,y: uay.update_plot(t,y), lambda t,x,y: uay.update_plot(t,y)))

/*Parametrized
       
Considering each time step individually is often infeasible

large parameter space
how to ensure smoothness?
how to achieve time invariance?
Parameterizing the trajectory using a fixed number  𝐵  of basis functions

𝝉=𝝓(𝑧)𝑤 
𝑧 : phase, i.e. (time)steps at which to compute  𝜏 
𝝓(𝑧) : basis, e.g., normalized radial:  𝜙𝑖(𝑧𝑡)=𝑏𝑖(𝑧𝑡)∑𝑗𝑏𝑗(𝑧𝑡) ,  𝑏𝑖=exp(−(𝑧𝑡−𝑐𝑖)22ℎ) 
𝑤 : weights*/

# Radial basis function
def radial_basis(phase,centers,bandwidth):
    bases = np.exp(-(np.repeat(phase[...,np.newaxis],centers.shape,-1)-centers)**2/(2*bandwidth)).T
    bases /= bases.sum(axis=0)
    return bases.T
    
# The recorded data
data = doa.data[-1]

# interpolated data...
# usually not required since recorded data
# comes in certain frequency!!    
hz = 100
def interp_data(data,hz):
    t = np.arange(data[0,0],data[0,-1],1/hz)
    x = np.interp(t,data[0,:],data[1,:])
    y = np.interp(t,data[0,:],data[2,:])
    return np.array([t,x,y]) 

# Number of bases
B = 15

# The timesteps for the basis functions
z = np.linspace(data[0,0],data[0,-1],100)

# Equidistant centers for the bases
c = np.linspace(z[0],z[-1],B)

# A heursitic for a possible/plausible bandwidth
h = -(c[1]-c[0])**2/(2*np.log(0.3))

ϕ = radial_basis(z,c,h)
print("ϕ.shape: {}".format(ϕ.shape))
w = np.ones(B)


doa.clean()
doa.ax.plot(data[1,:],data[2,:],'b-o',linewidth=3)

uax.clean()
uax.ax.plot(data[0,:],data[1,:],'b-o',linewidth=3)
uax.ax.plot(z,ϕ*w,linewidth=3)
uax.ax.plot(z,np.dot(ϕ,w),'g',linewidth=3)


uay.clean()
uay.ax.plot(data[0,:],data[2,:],'b-o',linewidth=3)
uay.ax.plot(z,ϕ*w,linewidth=3)
uay.ax.plot(z,np.dot(ϕ,w),'g',linewidth=3)

/*Learning the weights  𝑤 
straight forward approach learn the weights via ridge regression --  𝑤=(𝝓(𝑧)𝑇𝝓(𝑧)+𝜆𝐼)−1𝝓(𝑧)𝑇𝝉*/


z = data[0,:]
ϕ = radial_basis(z,c,h)
    

# ridge regression
def learn_weights_1(data, ϕ, λ=1e-6):
    wx = np.dot(np.linalg.inv(np.dot(ϕ.T,ϕ)+λ*np.eye(B)),np.dot(ϕ.T,data[1,:]))
    wy = np.dot(np.linalg.inv(np.dot(ϕ.T,ϕ)+λ*np.eye(B)),np.dot(ϕ.T,data[2,:]))
    return np.array([wx,wy])

w1 = learn_weights_1(data, ϕ)
# also ridge regression but numerically more stable
def learn_weights_2(data, ϕ, λ=1e-6):
    wx = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[1,:]))
    wy = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[2,:]))
    return np.array([wx,wy])

w2 = learn_weights_2(data, ϕ)

print("max abs difference between inv and solve: {}".format(np.max(np.abs(w1-w2))))

# still ridge regression but for all dimensions at once
def learn_weights(data, ϕ, λ=1e-6):
    w = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[1:,:].T)).T
    return w

w = learn_weights(data, ϕ)

print("w.shape: {}".format(w.shape))
print("max abs difference between solve and solve for all dims: {}".format(np.max(np.abs(w-w2))))

uax.clean()
uax.ax.plot(data[0,:],data[1,:],'r-x',linewidth=3)
uax.ax.plot(z,ϕ*w[0,:],linewidth=3)
uax.ax.plot(z,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)


uay.clean()
uay.ax.plot(data[0,:],data[2,:],'r-x',linewidth=3)
uay.ax.plot(z,ϕ*w[1,:],linewidth=3)
uay.ax.plot(z,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)


doa.clean()
doa.ax.plot(data[1,:],data[2,:],'r-x',linewidth=3)
doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)

##############
z = np.arange(data[0,0],data[0,-1],1/10)

ϕ = radial_basis(z,c,h)

uax.clean()
uax.ax.plot(z,ϕ*w[0,:],linewidth=3)
uax.ax.plot(z,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)


uay.clean()
uay.ax.plot(z,ϕ*w[1,:],linewidth=3)
uay.ax.plot(z,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)


doa.clean()
doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)

/*Time Invariance
How can we execute a learned trajectory faster or slower?
extend phase  𝑧 
recompute centers  𝑐 
set new bandwidth  ℎ 
recompute basis*/

des_duration = 2

z = np.arange(data[0,0],des_duration,1/hz)
c = np.linspace(z[0],z[-1],B)
h = -(c[1]-c[0])**2/(2*np.log(0.3))

ϕ = radial_basis(z,c,h)

uax.clean()
uax.ax.plot(z,ϕ*w[0,:],linewidth=3)
uax.ax.plot(z,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)


uay.clean()
uay.ax.plot(z,ϕ*w[1,:],linewidth=3)
uay.ax.plot(z,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)


doa.clean()
doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)

/*Simpler alternative:
temporal scaling of each trajectory, such that it was always executed between  0.0  and  1.0 
centers never have to change
bandwidth never has to change
phase is just normalized time*/

def get_phase(t):
    phase = t-np.min(t)
    phase /= np.max(phase)
    return phase

# Equidistant centers for the bases
c = np.linspace(0.0,1.0,B)
# A heursitic for a possible/plausible bandwidth
h = -(1/(B-1))**2/(2*np.log(0.3))

z = get_phase(data[0,:])
ϕ = radial_basis(z,c,h)

w = learn_weights(data,ϕ)

uax.clean()
uax.ax.plot(data[0,:],data[1,:],'r-x',linewidth=3)
uax.ax.plot(data[0,:],ϕ*w[0,:],linewidth=3)
uax.ax.plot(data[0,:],np.dot(ϕ,w[0,:]),'g-s',linewidth=3)


uay.clean()
uay.ax.plot(data[0,:],data[2,:],'r-x',linewidth=3)
uay.ax.plot(data[0,:],ϕ*w[1,:],linewidth=3)
uay.ax.plot(data[0,:],np.dot(ϕ,w[1,:]),'g-s',linewidth=3)


doa.clean()
doa.ax.plot(data[1,:],data[2,:],'r-x',linewidth=3)
doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)

##########

des_t = np.arange(0.0,des_duration,1/hz)

z = get_phase(des_t)
ϕ = radial_basis(z,c,h)


uax.clean()
uax.ax.plot(data[0,:],data[1,:],'r-x',linewidth=3)
uax.ax.plot(des_t,ϕ*w[0,:],linewidth=3)
uax.ax.plot(des_t,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)


uay.clean()
uay.ax.plot(data[0,:],data[2,:],'r-x',linewidth=3)
uay.ax.plot(des_t,ϕ*w[1,:],linewidth=3)
uay.ax.plot(des_t,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)


doa.clean()
doa.ax.plot(data[1,:],data[2,:],'r-x',linewidth=3)
doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)

/*Considering Multiple Demonstrations
We consider each demonstration as a 'variation' or 'sample' of the same movement primitive
hence, we treat the weight vector for each demonstration  𝑤𝑖  as an instance of a random variable, drawn from a multivariate Gaussian  𝑤𝑖∼(𝑤|𝜇𝑤,Σ𝑤)*/

doa.clear()
uax.clear()
uay.clear()


# Equidistant centers for the bases
c = np.linspace(0.0,1.0,B)
# A heursitic for a possible/plausible bandwidth
h = -(1/(B-1))**2/(2*np.log(0.3))

trajectories = [interp_data(d,hz) for d in doa.data]

def learn_weight_distribution(trajectories):
    ws = np.array([learn_weights(d,radial_basis(get_phase(d[0,:]),c,h)).flatten() for d in trajectories])
    μ = np.mean(ws,axis=0)
    Σ = np.cov(ws.T)
    return μ, Σ

μ_w, Σ_w = learn_weight_distribution(trajectories)

#We can now sample from the weight distribution and produce new similar trajectories

doa.clean()
uax.clean()
uay.clean()

des_duration = 2
des_t = np.arange(0.0,des_duration,1/hz)

z = get_phase(des_t)
ϕ = radial_basis(z,c,h)

ws = np.random.multivariate_normal(μ_w, Σ_w, 10)
ws = ws.reshape(ws.shape[0],-1,B)
print(ws.shape)
uax.ax.plot(des_t,np.dot(ϕ,ws[:,0,:].T),linewidth=3)
uay.ax.plot(des_t,np.dot(ϕ,ws[:,1,:].T),linewidth=3)
doa.ax.plot(np.dot(ϕ,ws[:,0,:].T),np.dot(ϕ,ws[:,1,:].T),linewidth=3)

/*We can even write down the distribution in the original trajectory space:

𝝉∼(𝝉|𝜇𝜏,Σ𝜏) 
𝜇𝜏=Ψ𝜇𝑤 
Σ𝜏=ΨΣ𝑤Ψ𝑇+Σobs 
Ψ : block diagonal of  𝐷  blocks. Each block being  𝜙*/



def get_traj_distribution(μ_w, Σ_w, des_duration=1.0):
    des_t = np.arange(0.0,des_duration,1/hz)
    z = get_phase(des_t)
    ϕ = radial_basis(z,c,h)
    D = 2
    Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)
    μ = np.dot(Ψ,μ_w)
    Σ = np.dot(np.dot(Ψ,Σ_w),Ψ.T)
    return μ, Σ

μ_τ, Σ_τ = get_traj_distribution(μ_w, Σ_w, des_duration)


des_t = np.arange(0.0,des_duration,1/hz)
μ_D = μ_τ.reshape((-1,des_t.shape[0]))

uax.ax.plot(des_t,μ_D[0,:],'m',linewidth=5)
uay.ax.plot(des_t,μ_D[1,:],'m',linewidth=5)

σ_τ = np.sqrt(np.diag(Σ_τ))
σ_D = σ_τ.reshape((-1,des_t.shape[0]))

uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='m',alpha=0.3)
uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='m',alpha=0.3)

#####

#implemented function, might still have to mess with matrices dimensions to work
def conditioning(μ_w, Σ_w, y_t, Σ_y):
    des_t = np.arange(0.0,des_duration,1/hz)
    z = get_phase(des_t)
    ϕ = radial_basis(z,c,h)
    D = 2
    Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)
    
    L = Σ_w.dot(Ψ).dot(inv(Σ_y + np.transpose(Ψ).dot(Σ_w).dot(Ψ)))
    new_μ_w = μ_w + L.dot(y_t - np.transpose(Ψ).dot(μ_w))
    new_Σ_w = Σ_w - Lnp.dot(np.transpose(Ψ)).dot(Σ_w)
    
    return new_μ_w, new_Σ_w

#call to function
conditioned_μ_w, conditioned_Σ_w = conditioning(μ_w, Σ_w, )

#final steps are repeated again to create trajectory and plot it on graph
μ_τ, Σ_τ = get_traj_distribution(conditioned_μ_w, conditioned_Σ_w, des_duration)


des_t = np.arange(0.0,des_duration,1/hz)
μ_D = μ_τ.reshape((-1,des_t.shape[0]))

uax.ax.plot(des_t,μ_D[0,:],'m',linewidth=5)
uay.ax.plot(des_t,μ_D[1,:],'m',linewidth=5)

σ_τ = np.sqrt(np.diag(Σ_τ))
σ_D = σ_τ.reshape((-1,des_t.shape[0]))

uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='m',alpha=0.3)
uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='m',alpha=0.3)




