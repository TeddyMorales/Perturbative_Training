import matplotlib.pyplot as plt
import time
import numpy as np


#ridge regression to fit weights
def learn_weights(data, ϕ, λ=1e-6):
    w = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[1:,:].T)).T
    return w

#function to interpolate data or to fix its frequency/resolution (filtering)
def interp_data(data,hz):
    t = np.arange(data[0,0],data[0,-1],1/hz)
    x = np.interp(t,data[0,:],data[1,:])
    y = np.interp(t,data[0,:],data[2,:])
    return np.array([t,x,y])

#creates the basis functions and returns them
def radial_basis(phase,centers,bandwidth):
    bases = np.exp(-(np.repeat(phase[...,np.newaxis],centers.shape,-1)-centers)**2/(2*bandwidth)).T
    bases /= bases.sum(axis=0)
    return bases.T

#temporal scaling of each trajectory, such that it was always executed between  0.0  and  1.0
#centers never have to change
#bandwidth never has to change
#phase is just normalized time
def get_phase(t):
    phase = t-np.min(t)
    phase /= np.max(phase)
    return phase

#function to train weigth sfor multiple dimension
def learn_weight_distribution(trajectories, z, ϕ):
    #learns weights for each dimension using time invariant phases
    ws = np.array([learn_weights(d, ϕ).flatten() for d in trajectories])
    μ = np.mean(ws,axis=0)
    #covariance matrix
    Σ = np.cov(ws.T)
    return μ, Σ

#function for trajectory distribution
def get_traj_distribution(μ_w, Σ_w, des_duration=1.0):
    des_t = np.arange(0.0,des_duration,1/hz)
    z = get_phase(des_t)
    ϕ = radial_basis(z,c,h)
    D = 2
    Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)
    
    μ = np.dot(Ψ,μ_w)
    Σ = np.dot(np.dot(Ψ,Σ_w),Ψ.T)
    return μ, Σ

#implemented function, might still have to mess with matrices dimensions to work
def conditioning(μ_w, Σ_w, y_t, Σ_y, Ψ):
    inverse = np.linalg.inv(Σ_y + np.dot(np.dot(Ψ.T, Σ_w), Ψ))
    L = np.dot(np.dot(Σ_w, Ψ), inverse)
    
    new_μ_w = μ_w + np.dot(L, y_t - np.dot(Ψ.T, μ_w))
    new_Σ_w = Σ_w - np.dot(np.dot(L, Ψ.T), Σ_w)
    return new_μ_w, new_Σ_w

def get_Ψ(z, dims, D, B):
    z = np.array(z,ndmin=1)
    length = z.shape[0]
    c = np.linspace(0.0,1.0,B)
    h = -(1/(B-1))**2/(2*np.log(0.3))
    
    basis = radial_basis(z,c,h)
    Ψ = np.zeros((D * B, len(dims) * length))
    for idx, d in enumerate(dims):
        Ψ[d * B : (d+1) * B, idx * length : (idx + 1) * length] = basis.T
    return Ψ

#the data we will use
#data = 

hz = 100

#number of base functions
B = 15

#number of dimensions
D = 2

#z is the phase, can be extended over the data the other over the desired des_t amount of time
z = get_phase(data[0,:])

# Equidistant centers for the bases
c = np.linspace(0.0,1.0,B)

# A heursitic for a possible/plausible bandwidth
h = -(1/(B-1))**2/(2*np.log(0.3))

#basis functions, equidistant centers, 
ϕ = radial_basis(z,c,h)


#learning the weights for multiple dimensions

#interpolates data d being the individual dimensions of data
trajectories = [interp_data(d,hz) for d in doa.data]

#to learn weights over dimensions using the intrpolated data of each dimension, time invariant phases, basis functions
μ_w, Σ_w = learn_weight_distribution(trajectories, z, ϕ)

#computing trajectories ovedesired time variable
des_duration = 2
des_t = np.arange(0.0,des_duration,1/hz)

#recalculate phase and basis functions
z = get_phase(des_t)
ϕ = radial_basis(z,c,h)

#Draw random samples from a multivariate normal distribution using the mean (μ_w) and covariance matrix (Σ_w) 
ws = np.random.multivariate_normal(μ_w, Σ_w, 10)
#reshapes ws to numpy array that contains data for all new random samples
ws = ws.reshape(ws.shape[0],-1,B)

#finds distribution of new trajectory samples, returns distribution and mean trajectory
μ_τ, Σ_τ = get_traj_distribution(μ_w, Σ_w, des_duration)

#reshapes mean trajectory of distribution to BE plottable
des_t = np.arange(0.0,des_duration,1/hz)
μ_D = μ_τ.reshape((-1,des_t.shape[0]))
uax.ax.plot(des_t,μ_D[0,:],'m',linewidth=5)
uay.ax.plot(des_t,μ_D[1,:],'m',linewidth=5)


σ_τ = np.sqrt(np.diag(Σ_τ))
σ_D = σ_τ.reshape((-1,des_t.shape[0]))
uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='m',alpha=0.3)
uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='m',alpha=0.3)


#which dimensions conditioned on [x, y, z]
dims = [0, 1]
#put value between 0 and 1 for timestep conditioned on (can condition on more than one)
z = [.25]
#put values at [x, y, z] (must match dims)
features = [.4, .7]
#how much slack you give the distribution
Σ = .001

#c_Ψ = R. D* times #L X D time B
c_Ψ = get_Ψ(z, dims, D, B)

#y* is value of x and y we want to condition to in vector form
#Σ* is a D X D matrix (covariance matrix)
y_t = np.array(features)
Σ_y = np.diag([Σ]*len(y_t))

#call to function
c_μ_τ, c_Σ_τ = conditioning(μ_w, Σ_w, y_t, Σ_y, c_Ψ)

#get new mean with conditioning
μ_τ, Σ_τ = get_traj_distribution(c_μ_τ, c_Σ_τ, des_duration)

des_t = np.arange(0.0,des_duration,1/hz)
μ_D = μ_τ.reshape((-1,des_t.shape[0]))
uax.ax.plot(des_t,μ_D[0,:],'b',linewidth=5)
uay.ax.plot(des_t,μ_D[1,:],'b',linewidth=5)


σ_τ = np.sqrt(np.diag(Σ_τ))
σ_D = σ_τ.reshape((-1,des_t.shape[0]))
uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='b',alpha=0.3)
uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='b',alpha=0.3)


