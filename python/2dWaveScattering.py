#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:04:08 2019

@author: jrtroy

This project simulates the 2 d scattering of a wave source off a 
reflector disk in simulation of radar and accoustic antennae.
"""

# setup and parameters
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib import cm

# plotting color mapping
cmapName = "viridis"

# simulation parameters
eps = 2.0/10        # size of initial condition 
tol = 1e-3          # allowed tolerances
videoInterval = 1e-4    # ms


# spacing

X = 2
Y = 2
xspan = [-X,X]
yspan = [-Y,Y]
tspan = [0,2]


# x and y nodes 
Mx = 300
My = 300
N = 300        # t nodes 

# construction of spacial grid
x = np.linspace(xspan[0],xspan[1],Mx)
y = np.linspace(yspan[0],yspan[1],My)
[xmat,ymat] = np.meshgrid(x,y)

space = [xmat,ymat]

# step sizes and approximate wave speed
hx = x[1] - x[0]
hy = y[1] - y[0]
tau = (tspan[1] - tspan[0])/N

waveSpeed = np.sqrt((hx**2 + hy**2)/(2*tau**2))

print("Wave speed = " + str(waveSpeed))



# functions

# initial condition
def initialCondition(space,size):
    """
    Initial condition for wave simulation
    Input: 
        space = [xmat, ymat]    list of space matrices
        size    (scalar)        size of the initial spike
    Output:
        array of same dimensions of space - one for initial condition,
        and one for the initial 'velocities'
        Of form     (1/e^2) (|x|^2 - e^2)^2
    """
    xmat = space[0]
    ymat = space[1]
    
    # functional construction
    psiInit = np.power(xmat,2) + np.power(ymat,2) - np.power(size,2)
    psiInit = np.divide(np.power(psiInit,2),np.power(size,4))
    
    # enforce is zero outside of range
    psiInit[np.power(xmat,2) + np.power(ymat,2) > size**2] = 0
    
    # 'velocities'
    psiInitPrime = np.zeros(np.shape(psiInit))
    
    return [psiInit,psiInitPrime]

# differentiation matrices
def buildDiff2MatOrder2(x):
    """Building the second order differentiation matrix Dxx
    Input: 
        x   vector (length m) of x coordinates
    Output:
        m x m matrix for second derivative
    """
    size = len(x)
    h = x[1] - x[0]
    
    # construction through diagonals
    Dxx = np.diagflat(-2*np.ones([size,1]))
    Dxx += np.diagflat(np.ones([size-1,1]),-1) + np.diagflat(np.ones([size-1,1]),1)
    
    # fixing first and last rows
    Dxx[0,:4] = [1, -2.5, 2, -0.5]
    Dxx[-1,-4:] = [0.5, -2, 2.5, -1]
    
    # scaling
    Dxx *= 2/np.power(h,2)
    return Dxx

# second derivative of psi with respect to time
def secondDerivative(t,psi,params):
    """
    Constructing the second derivative with respect to time
    This function will be used in the integration method
    Input:
        t       time (vector or scalar)
        psi     current value of psi
        params  list of miscillaneous parameters:
            params = [
                    Dxx
                    Dyy     differentiation matrices for x and y
                    arcfun  callable function describing location of reflector
                        should be of the form f(x,y) = 0
                    [xmat,ymat]     input for arcfun
                    tol     tolerance value for arcfun
                    ]
    Output:
        Laplacian of psi
    """
    # unpack params
    Dxx = params[0]
    Dyy = params[1]
    arcfun = params[2]
    x = params[3]
    tol = params[4]
    
    # reflection
    psi[np.abs(reflectorLocation(x,arcfun)) < tol] = 0
    
    # compute laplacian
    laplacian = np.matmul(Dxx,psi) + np.matmul(psi,Dyy.T)
    
    return laplacian
    
# interation method
def verlet(fun,tspan,psiInit,params):
    """
    Verlet integration for ODE
    Input:
        fun     callable function computing d^2/dt^2 psi
        tspan   [t0,t0+T]   span for time
        psiInit Initial conditions
        params  miscilaneous parameters
            params = [
                    [Mx,My,N]   spacing for space and time
                    funParams   parameters for second derivative function
                    ]
    Output:
        t - time values for evaluation and 
        Large array of psi values at all x values and t values
    """
    
    # unpacking
    Mx = params[0][0]
    My = params[0][1]
    N = params[0][2]
    funParams = params[1]
    
    # construct time vector
    t = np.linspace(tspan[0],tspan[1],N)
    tau = t[1] - t[0]       # time spacing
    
    # dimensions for psi array
    dims = np.shape(psiInit[0])
    dims = np.insert(dims,0,N)
    
    # allocate space for psi
    psi = np.zeros(dims)
    psi[0,:,:] = psiInit[0]         # initial condition
    
    # starting step
    psi[1,:,:] = psiInit[0] + tau * psiInit[1] + np.power(tau,2)/2 * fun(t,psiInit[0],funParams)
    
    # main iteration
    for j in range(1,len(t)-1):
        psi[j+1,:,:] = 2*psi[j,:,:] - psi[j-1,:,:] + np.power(tau,2) * fun(t,psi[j,:,:],funParams)
    
    return [t,psi]
    
def reflectorLocation(x,locationFun):
    """
    function describing location and shape of reflector
    f(x) = 0, note x is a vector
    Input:
        x - vector of x and y coordinates
        locationFun - function dictating location
            of form f(x,y) = 0
    Output:
        f(x)
    """
    # unpacking
    xmat = x[0]
    ymat = x[1]
    
    lam = 0.5
    
    # construction of function
    
    # semicircle
    #tmp = np.power(xmat,2) + np.power(ymat,2) - np.power(lam,2)
    
    # parabola
    #tmp = ymat - (np.power(xmat,2)/(2*lam) - lam/2)
    
    # ellipse
    #alpha = 3           # scale parameter for ellipse
    #a = lam**2 * (1 + alpha)
    #b = (lam**2 + a**2)/(2*lam)
    
    #tmp = 1/a**2 * np.power(xmat,2) + 1/b**2 * np.power(ymat - np.sqrt(b**2 - a**2),2) - 1
    #tmp[ymat > 0] = 1          # ensuring only occurs for y < 0
    
    levels = locationFun(xmat,ymat)
    levels[ymat > 0] = 10
    return levels

# semicircle reflector
def semiCircleReflector(x,y,radius=0.5):
    levels =  np.power(x,2) + np.power(y,2) - np.power(radius,2)
    return levels

# parabolic reflector
def parabolicReflector(x,y,lam=0.5):
    levels = y - (np.power(x,2)/(2 * lam) - lam/2)
    return levels

# elliptic reflector
def ellipticReflector(x,y,lam=0.5,alpha=3):
    a = lam**2 * (1 + alpha)
    b = (lam**2 + a**2)/(2 * lam)
    
    levels = 1/a**2 * np.power(x,2) + 1/b**2 * np.power(y,2) - 1
    return levels

# calculating efficiency


# script

# initial conditions
psiInit = initialCondition(space,eps)

# building parameters

# differentiation matrices
Dxx = buildDiff2MatOrder2(x)
Dyy = buildDiff2MatOrder2(y)

discreteParams = [Mx,My,N]
derivativeParams = [Dxx,Dyy,ellipticReflector,space,tol]
mainParams = [discreteParams,derivativeParams]

[t,psi] = verlet(secondDerivative,tspan,psiInit,mainParams)

fig,ax = plt.subplots()
ax.contour(xmat,ymat,reflectorLocation(space,ellipticReflector))
ax.set_aspect("equal","box")

# plotting
fig,ax = plt.subplots()

normalization = cm.colors.Normalize(vmax=0.25,vmin=0)

# animation
def animate(i):
    ax.clear()
    ax.contourf(xmat,ymat,np.abs(psi[i,:,:]),norm=normalization,cmap=plt.get_cmap(cmapName))
    ax.set_aspect("equal","box")

def generateAnimation(fileName):
    anim = ani.FuncAnimation(fig,animate,N,interval=videoInterval*1e+3,blit=False,repeat=True)
    anim.save("test.gif", writer="imagemagick")

# initial condition
fig,ax = plt.subplots()
ax.contourf(xmat,ymat,psi[0,:,:],norm=normalization,cmap=plt.get_cmap(cmapName))
ax.set_aspect("equal","box")
ax.set_title("Initial Condition")


generateAnimation("test.gif")