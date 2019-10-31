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
def get_traj_distribution(μ_w, Σ_w, z, ϕ, B):
    D = 2
    Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)
    μ = np.dot(Ψ,μ_w)
    Σ = np.dot(np.dot(Ψ,Σ_w),Ψ.T)
    return μ, Σ

def conditioning(μ_w, Σ_w, y_t, Σ_y, Ψ):
    
    L = (Σ_w.dot(Ψ)).dot(np.linalg.inv(Σ_y + (Ψ.T).dot(Σ_w).dot(Ψ)))
    
    new_μ_w = μ_w + L.dot(y_t - (Ψ.T).dot(μ_w))
    new_Σ_w = Σ_w - L.dot(Ψ.T).dot(Σ_w)
    return new_μ_w, new_Σ_w



#the data we will use
#data = 

hz = 100

#number of base functions
B = 15

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

#computing trajectories ovesired time variable
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
μ_τ, Σ_τ = get_traj_distribution(μ_w, Σ_w, z, ϕ, B)

#reshapes mean trajectory of distribution to BE plottable
μ_D = μ_τ.reshape((-1,des_t.shape[0]))


σ_τ = np.sqrt(np.diag(Σ_τ))
σ_D = σ_τ.reshape((-1,des_t.shape[0]))




Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)

#call to function
μ_τ, Σ_τ = conditioning(μ_τ, Σ_τ, 0.5, .1, Ψ)


#Ψ = np.kron(np.eye(int(μ_τ.shape[0]/B),dtype=int),ϕ)

#get new mean with conditioning
#μ_τ = np.dot(Ψ.T,μ_τ)
#get new distribution with conditioning
#Σ_τ = np.dot(np.dot(Ψ,Σ_τ),Ψ.T)

print(μ_τ.shape)
des_t = np.arange(0.0,des_duration,1/hz)
μ_D = μ_τ.reshape((-1,des_t.shape[0]))
print(μ_D.shape)

uax.ax.plot(des_t,μ_D[0,:],'m',linewidth=5)
uay.ax.plot(des_t,μ_D[1,:],'m',linewidth=5)

#print(np.diag(Σ_τ).shape)
#σ_τ = np.sqrt(np.diag(Σ_τ))
#σ_D = σ_τ.reshape((-1,des_t.shape[0]))

#uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='m',alpha=0.3)
#uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='m',alpha=0.3)


