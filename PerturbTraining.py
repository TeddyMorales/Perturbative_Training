{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pybullet as p \n",
    "import time \n",
    "import pybullet_data \n",
    "import random\n",
    "\n",
    "class huge_robot(object):\n",
    "\n",
    "\n",
    "    def __init__(self):\n",
    "        physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version\n",
    "        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optional \n",
    "\n",
    "        pass\n",
    "\n",
    "    def pertubation(self, n, m, enviro_vars):\n",
    "        '''performs pertubative training by changing variables randomly \n",
    "            within a range for each variables parameters  '''\n",
    "        for (i in range n):\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib tk\n",
    "import matplotlib.pyplot as plt\n",
    "import time\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class AxUpdater(object):\n",
    "    \n",
    "    def __init__(self, fig, ax, xlim=None, ylim=None):\n",
    "        \n",
    "        self.fig = fig\n",
    "        self.ax = ax\n",
    "        \n",
    "        self.xlim = xlim\n",
    "        if xlim is not None:\n",
    "            self.ax.set_xlim(xlim[0],xlim[1])\n",
    "        \n",
    "        self.ylim = ylim\n",
    "        if ylim is not None:\n",
    "            self.ax.set_ylim(ylim[0],ylim[1])\n",
    "        \n",
    "        self.trajs = []\n",
    "        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)\n",
    "\n",
    "    def new_plot(self):\n",
    "        self.trajs.append(self.ax.plot([],[], '-o')[0])\n",
    "        \n",
    "        \n",
    "    def update_plot(self,x=None,y=None):\n",
    "        assert(len(self.trajs))\n",
    "        if x is not None and y is not None:\n",
    "            self.trajs[-1].set_data(x, y)\n",
    "        \n",
    "        self.fig.canvas.restore_region(self.background)\n",
    "        # redraw just the points\n",
    "        self.ax.draw_artist(self.trajs[-1])\n",
    "        # fill in the axes rectangle\n",
    "        self.fig.canvas.blit(self.ax.bbox)\n",
    "        \n",
    "    def clean(self):\n",
    "        for idx in range(len(self.ax.lines)-1,-1,-1):\n",
    "            if self.ax.lines[idx] not in self.trajs:\n",
    "                self.ax.lines.pop(idx)\n",
    "        for idx in range(len(self.ax.collections)-1,-1,-1):\n",
    "            self.ax.collections.pop(idx)\n",
    "        self.fig.canvas.draw()\n",
    "        self.fig.canvas.flush_events()\n",
    "        \n",
    "    def clear(self):\n",
    "        self.ax.set_prop_cycle(None)\n",
    "        self.trajs=[]\n",
    "        self.clean()\n",
    "\n",
    "class DrawOnAx(AxUpdater):\n",
    "\n",
    "    def __init__(self, fig, ax, xlim=(0.0,1.0),ylim=(0.0,1.0)):\n",
    "        AxUpdater.__init__(self, fig, ax, xlim, ylim)\n",
    "        \n",
    "        self.curr_x = None\n",
    "        self.curr_y = None\n",
    "        self.curr_t = None\n",
    "        self.curr_t_start = None\n",
    "        self.data = []\n",
    "        self.cbs = []\n",
    "\n",
    "        self.do_record = False\n",
    "        \n",
    "        self.fig.canvas.mpl_connect(\"button_press_event\", self.on_press)\n",
    "        self.fig.canvas.mpl_connect(\"button_release_event\", self.on_release)\n",
    "        self.fig.canvas.mpl_connect(\"motion_notify_event\", self.on_move)\n",
    "\n",
    "    def add_cb(self,cb):\n",
    "        assert(isinstance(cb,tuple) and len(cb)==3)\n",
    "        \n",
    "        self.cbs.append(cb)\n",
    "        \n",
    "    def on_press(self, event):\n",
    "        if event.inaxes != self.ax:\n",
    "            return\n",
    "        if self.do_record:\n",
    "            return\n",
    "        self.curr_t_start = time.time() \n",
    "        self.curr_t = [0.0]\n",
    "        self.curr_x = [event.xdata]\n",
    "        self.curr_y = [event.ydata]\n",
    "        self.new_plot()\n",
    "        self.update_plot(self.curr_x,self.curr_y)\n",
    "        self.do_record=True\n",
    "        \n",
    "        for cb in self.cbs:\n",
    "            if cb[0] is not None:\n",
    "                cb[0](self.curr_t,self.curr_x,self.curr_y)\n",
    "\n",
    "    def on_move(self, event):\n",
    "        if event.inaxes != self.ax:\n",
    "            return\n",
    "        if not self.do_record:\n",
    "            return\n",
    "        \n",
    "        self.curr_t.append(time.time()-self.curr_t_start)\n",
    "        self.curr_x.append(event.xdata)\n",
    "        self.curr_y.append(event.ydata)\n",
    "        self.update_plot(self.curr_x,self.curr_y)\n",
    "        \n",
    "        for cb in self.cbs:\n",
    "            if cb[1] is not None:\n",
    "                cb[1](self.curr_t,self.curr_x,self.curr_y)\n",
    "\n",
    "    def on_release(self, event):\n",
    "        if event.inaxes != self.ax:\n",
    "            return\n",
    "        if not self.do_record:\n",
    "            return\n",
    "        self.do_record = False\n",
    "        self.curr_t.append(time.time()-self.curr_t_start)\n",
    "        self.curr_x.append(event.xdata)\n",
    "        self.curr_y.append(event.ydata)\n",
    "        self.update_plot(self.curr_x,self.curr_y)\n",
    "        self.data.append(np.array([self.curr_t, self.curr_x, self.curr_y]))\n",
    "        \n",
    "        for cb in self.cbs:\n",
    "            if cb[2] is not None:\n",
    "                cb[2](self.curr_t,self.curr_x,self.curr_y)\n",
    "\n",
    "\n",
    "    def clear(self):\n",
    "        self.data = []\n",
    "        AxUpdater.clear(self)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Motivation\n",
    "$\\newcommand\\traj{\\boldsymbol{\\tau}}$\n",
    "Given:\n",
    "- Trajectory $\\traj \\in \\mathbb{R}^{D\\times L}$, with dimension $D$, over $L$ time steps\n",
    "\n",
    "Desired: Movement representation that\n",
    "- is parametrized\n",
    "- is time invariant\n",
    "- considers multiple demonstrations\n",
    "- captures correlations between DoFs\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axs = plt.subplots(3,1,squeeze=False)\n",
    "\n",
    "doa = DrawOnAx(fig, axs[0][0])\n",
    "uax = AxUpdater(fig, axs[1][0],xlim=(0.0,3.0),ylim=(0.0,1.0))\n",
    "uay = AxUpdater(fig, axs[2][0],xlim=(0.0,3.0),ylim=(0.0,1.0))\n",
    "\n",
    "doa.add_cb((lambda t,x,y: (uax.new_plot(),uax.update_plot(t,x)), lambda t,x,y: uax.update_plot(t,x), lambda t,x,y: uax.update_plot(t,x)))\n",
    "doa.add_cb((lambda t,x,y: (uay.new_plot(),uay.update_plot(t,y)), lambda t,x,y: uay.update_plot(t,y), lambda t,x,y: uay.update_plot(t,y)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Parametrized\n",
    "$\\newcommand\\basis{\\boldsymbol{\\phi}}$\n",
    "$\\newcommand\\w{\\boldsymbol{w}}$\n",
    "$\\newcommand\\phase{\\boldsymbol{z}}$\n",
    "- Considering each time step individually is often infeasible\n",
    "    - large parameter space\n",
    "    - how to ensure smoothness?\n",
    "    - how to achieve time invariance?\n",
    "    \n",
    "- Parameterizing the trajectory using a fixed number $B$ of basis functions\n",
    "    - $\\traj = \\basis(\\phase)\\w$\n",
    "        - $\\phase$: phase, i.e. (time)steps at which to compute $\\tau$\n",
    "        - $\\basis(\\phase)$: basis, e.g., normalized radial: $\\phi_i(z_t) = \\dfrac{b_i(z_t)}{\\sum_j b_j(z_t)}$, $b_i=\\exp\\left(-\\dfrac{(z_t-c_i)^2}{2h}\\right)$\n",
    "        - $\\w$: weights\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ϕ.shape: (100, 15)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1193414a8>]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Radial basis function\n",
    "def radial_basis(phase,centers,bandwidth):\n",
    "    bases = np.exp(-(np.repeat(phase[...,np.newaxis],centers.shape,-1)-centers)**2/(2*bandwidth)).T\n",
    "    bases /= bases.sum(axis=0)\n",
    "    return bases.T\n",
    "    \n",
    "# The recorded data\n",
    "data = doa.data[-1]\n",
    "\n",
    "# interpolated data...\n",
    "# usually not required since recorded data\n",
    "# comes in certain frequency!!    \n",
    "hz = 100\n",
    "def interp_data(data,hz):\n",
    "    t = np.arange(data[0,0],data[0,-1],1/hz)\n",
    "    x = np.interp(t,data[0,:],data[1,:])\n",
    "    y = np.interp(t,data[0,:],data[2,:])\n",
    "    return np.array([t,x,y]) \n",
    "\n",
    "# Number of bases\n",
    "B = 15\n",
    "\n",
    "# The timesteps for the basis functions\n",
    "z = np.linspace(data[0,0],data[0,-1],100)\n",
    "\n",
    "# Equidistant centers for the bases\n",
    "c = np.linspace(z[0],z[-1],B)\n",
    "\n",
    "# A heursitic for a possible/plausible bandwidth\n",
    "h = -(c[1]-c[0])**2/(2*np.log(0.3))\n",
    "\n",
    "ϕ = radial_basis(z,c,h)\n",
    "print(\"ϕ.shape: {}\".format(ϕ.shape))\n",
    "w = np.ones(B)\n",
    "\n",
    "\n",
    "doa.clean()\n",
    "doa.ax.plot(data[1,:],data[2,:],'b-o',linewidth=3)\n",
    "\n",
    "uax.clean()\n",
    "uax.ax.plot(data[0,:],data[1,:],'b-o',linewidth=3)\n",
    "uax.ax.plot(z,ϕ*w,linewidth=3)\n",
    "uax.ax.plot(z,np.dot(ϕ,w),'g',linewidth=3)\n",
    "\n",
    "\n",
    "uay.clean()\n",
    "uay.ax.plot(data[0,:],data[2,:],'b-o',linewidth=3)\n",
    "uay.ax.plot(z,ϕ*w,linewidth=3)\n",
    "uay.ax.plot(z,np.dot(ϕ,w),'g',linewidth=3)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<span style=\"color:#AA0000; font-weight:bold;\">Obviously, something is wrong!</span>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>\n",
    "<br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Learning the weights $\\w$\n",
    "\n",
    "- straight forward approach learn the weights via ridge regression\n",
    "    -- $\\w = \\left(\\basis(\\phase)^T\\basis(\\phase) + \\lambda I\\right)^{-1} \\basis(\\phase)^T\\traj$\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "max abs difference between inv and solve: 3.1086244689504383e-15\n",
      "w.shape: (2, 15)\n",
      "max abs difference between solve and solve for all dims: 2.4424906541753444e-15\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x118f6a588>]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "z = data[0,:]\n",
    "ϕ = radial_basis(z,c,h)\n",
    "    \n",
    "\n",
    "# ridge regression\n",
    "def learn_weights_1(data, ϕ, λ=1e-6):\n",
    "    wx = np.dot(np.linalg.inv(np.dot(ϕ.T,ϕ)+λ*np.eye(B)),np.dot(ϕ.T,data[1,:]))\n",
    "    wy = np.dot(np.linalg.inv(np.dot(ϕ.T,ϕ)+λ*np.eye(B)),np.dot(ϕ.T,data[2,:]))\n",
    "    return np.array([wx,wy])\n",
    "\n",
    "w1 = learn_weights_1(data, ϕ)\n",
    "# also ridge regression but numerically more stable\n",
    "def learn_weights_2(data, ϕ, λ=1e-6):\n",
    "    wx = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[1,:]))\n",
    "    wy = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[2,:]))\n",
    "    return np.array([wx,wy])\n",
    "\n",
    "w2 = learn_weights_2(data, ϕ)\n",
    "\n",
    "print(\"max abs difference between inv and solve: {}\".format(np.max(np.abs(w1-w2))))\n",
    "\n",
    "# still ridge regression but for all dimensions at once\n",
    "def learn_weights(data, ϕ, λ=1e-6):\n",
    "    w = np.linalg.solve(np.dot(ϕ.T,ϕ)+λ*np.eye(B),np.dot(ϕ.T,data[1:,:].T)).T\n",
    "    return w\n",
    "\n",
    "w = learn_weights(data, ϕ)\n",
    "\n",
    "print(\"w.shape: {}\".format(w.shape))\n",
    "print(\"max abs difference between solve and solve for all dims: {}\".format(np.max(np.abs(w-w2))))\n",
    "\n",
    "uax.clean()\n",
    "uax.ax.plot(data[0,:],data[1,:],'r-x',linewidth=3)\n",
    "uax.ax.plot(z,ϕ*w[0,:],linewidth=3)\n",
    "uax.ax.plot(z,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "uay.clean()\n",
    "uay.ax.plot(data[0,:],data[2,:],'r-x',linewidth=3)\n",
    "uay.ax.plot(z,ϕ*w[1,:],linewidth=3)\n",
    "uay.ax.plot(z,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "doa.clean()\n",
    "doa.ax.plot(data[1,:],data[2,:],'r-x',linewidth=3)\n",
    "doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<span style='color:green; font-weight:bold;'>Because of the parametrization we can reproduce the trajectory in a different resolution</span>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x11931d278>]"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z = np.arange(data[0,0],data[0,-1],1/10)\n",
    "\n",
    "ϕ = radial_basis(z,c,h)\n",
    "\n",
    "uax.clean()\n",
    "uax.ax.plot(z,ϕ*w[0,:],linewidth=3)\n",
    "uax.ax.plot(z,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "uay.clean()\n",
    "uay.ax.plot(z,ϕ*w[1,:],linewidth=3)\n",
    "uay.ax.plot(z,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "doa.clean()\n",
    "doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Time Invariance\n",
    "\n",
    "- How can we execute a learned trajectory faster or slower?\n",
    "    1. extend phase $z$\n",
    "    2. recompute centers $c$\n",
    "    3. set new bandwidth $h$\n",
    "    4. recompute basis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1193256a0>]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "des_duration = 2\n",
    "\n",
    "z = np.arange(data[0,0],des_duration,1/hz)\n",
    "c = np.linspace(z[0],z[-1],B)\n",
    "h = -(c[1]-c[0])**2/(2*np.log(0.3))\n",
    "\n",
    "ϕ = radial_basis(z,c,h)\n",
    "\n",
    "uax.clean()\n",
    "uax.ax.plot(z,ϕ*w[0,:],linewidth=3)\n",
    "uax.ax.plot(z,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "uay.clean()\n",
    "uay.ax.plot(z,ϕ*w[1,:],linewidth=3)\n",
    "uay.ax.plot(z,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "doa.clean()\n",
    "doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Simpler alternative:\n",
    "- temporal scaling of each trajectory, such that it was always executed between $0.0$ and $1.0$\n",
    "    - centers never have to change\n",
    "    - bandwidth never has to change\n",
    "    - phase is just normalized time\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x11932d668>]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def get_phase(t):\n",
    "    phase = t-np.min(t)\n",
    "    phase /= np.max(phase)\n",
    "    return phase\n",
    "\n",
    "# Equidistant centers for the bases\n",
    "c = np.linspace(0.0,1.0,B)\n",
    "# A heursitic for a possible/plausible bandwidth\n",
    "h = -(1/(B-1))**2/(2*np.log(0.3))\n",
    "\n",
    "z = get_phase(data[0,:])\n",
    "ϕ = radial_basis(z,c,h)\n",
    "\n",
    "w = learn_weights(data,ϕ)\n",
    "\n",
    "uax.clean()\n",
    "uax.ax.plot(data[0,:],data[1,:],'r-x',linewidth=3)\n",
    "uax.ax.plot(data[0,:],ϕ*w[0,:],linewidth=3)\n",
    "uax.ax.plot(data[0,:],np.dot(ϕ,w[0,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "uay.clean()\n",
    "uay.ax.plot(data[0,:],data[2,:],'r-x',linewidth=3)\n",
    "uay.ax.plot(data[0,:],ϕ*w[1,:],linewidth=3)\n",
    "uay.ax.plot(data[0,:],np.dot(ϕ,w[1,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "doa.clean()\n",
    "doa.ax.plot(data[1,:],data[2,:],'r-x',linewidth=3)\n",
    "doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "reproductions in different velocities are now simpler\n",
    "    - compute phase of desired time axis\n",
    "    - recompute basis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x11930e390>]"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "des_t = np.arange(0.0,des_duration,1/hz)\n",
    "\n",
    "z = get_phase(des_t)\n",
    "ϕ = radial_basis(z,c,h)\n",
    "\n",
    "\n",
    "uax.clean()\n",
    "uax.ax.plot(data[0,:],data[1,:],'r-x',linewidth=3)\n",
    "uax.ax.plot(des_t,ϕ*w[0,:],linewidth=3)\n",
    "uax.ax.plot(des_t,np.dot(ϕ,w[0,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "uay.clean()\n",
    "uay.ax.plot(data[0,:],data[2,:],'r-x',linewidth=3)\n",
    "uay.ax.plot(des_t,ϕ*w[1,:],linewidth=3)\n",
    "uay.ax.plot(des_t,np.dot(ϕ,w[1,:]),'g-s',linewidth=3)\n",
    "\n",
    "\n",
    "doa.clean()\n",
    "doa.ax.plot(data[1,:],data[2,:],'r-x',linewidth=3)\n",
    "doa.ax.plot(np.dot(ϕ,w[0,:]),np.dot(ϕ,w[1,:]),'g-s',linewidth=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Defining a normalized phase also makes it easier to combine multiple demonstrations\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Considering Multiple Demonstrations\n",
    "\n",
    "- We consider each demonstration as a 'variation' or 'sample' of the same movement primitive\n",
    "- hence, we treat the weight vector for each demonstration $\\w_i$ as an instance of a random variable, drawn from a multivariate Gaussian $\\w_i \\sim \\mathcal{N}(\\w|\\mu_w,\\Sigma_w)$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "doa.clear()\n",
    "uax.clear()\n",
    "uay.clear()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Equidistant centers for the bases\n",
    "c = np.linspace(0.0,1.0,B)\n",
    "# A heursitic for a possible/plausible bandwidth\n",
    "h = -(1/(B-1))**2/(2*np.log(0.3))\n",
    "\n",
    "trajectories = [interp_data(d,hz) for d in doa.data]\n",
    "\n",
    "def learn_weight_distribution(trajectories):\n",
    "    ws = np.array([learn_weights(d,radial_basis(get_phase(d[0,:]),c,h)).flatten() for d in trajectories])\n",
    "    μ = np.mean(ws,axis=0)\n",
    "    Σ = np.cov(ws.T)\n",
    "    return μ, Σ\n",
    "\n",
    "μ_w, Σ_w = learn_weight_distribution(trajectories)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can now sample from the weight distribution and produce new similar trajectories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(10, 2, 15)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1191fe5f8>,\n",
       " <matplotlib.lines.Line2D at 0x11936f470>,\n",
       " <matplotlib.lines.Line2D at 0x11936f710>,\n",
       " <matplotlib.lines.Line2D at 0x11936f5c0>,\n",
       " <matplotlib.lines.Line2D at 0x11936f518>,\n",
       " <matplotlib.lines.Line2D at 0x11bbdd2e8>,\n",
       " <matplotlib.lines.Line2D at 0x1194e2390>,\n",
       " <matplotlib.lines.Line2D at 0x1194e25f8>,\n",
       " <matplotlib.lines.Line2D at 0x1194e24a8>,\n",
       " <matplotlib.lines.Line2D at 0x1194e2208>]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "doa.clean()\n",
    "uax.clean()\n",
    "uay.clean()\n",
    "\n",
    "des_duration = 2\n",
    "des_t = np.arange(0.0,des_duration,1/hz)\n",
    "\n",
    "z = get_phase(des_t)\n",
    "ϕ = radial_basis(z,c,h)\n",
    "\n",
    "ws = np.random.multivariate_normal(μ_w, Σ_w, 10)\n",
    "ws = ws.reshape(ws.shape[0],-1,B)\n",
    "print(ws.shape)\n",
    "uax.ax.plot(des_t,np.dot(ϕ,ws[:,0,:].T),linewidth=3)\n",
    "uay.ax.plot(des_t,np.dot(ϕ,ws[:,1,:].T),linewidth=3)\n",
    "doa.ax.plot(np.dot(ϕ,ws[:,0,:].T),np.dot(ϕ,ws[:,1,:].T),linewidth=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can even write down the distribution in the original trajectory space:\n",
    "- $\\traj \\sim \\mathcal{N}(\\traj|\\mu_\\tau,\\Sigma_\\tau)$\n",
    "    - $\\mu_\\tau = \\Psi\\mu_w$\n",
    "    - $\\Sigma_\\tau = \\Psi\\Sigma_w\\Psi^T+\\Sigma_{\\mathrm{obs}}$\n",
    "    - $\\Psi$: block diagonal of $D$ blocks. Each block being $\\phi$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PolyCollection at 0x119308208>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "def get_traj_distribution(μ_w, Σ_w, des_duration=1.0):\n",
    "    des_t = np.arange(0.0,des_duration,1/hz)\n",
    "    z = get_phase(des_t)\n",
    "    ϕ = radial_basis(z,c,h)\n",
    "    D = 2\n",
    "    Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)\n",
    "    μ = np.dot(Ψ,μ_w)\n",
    "    Σ = np.dot(np.dot(Ψ,Σ_w),Ψ.T)\n",
    "    return μ, Σ\n",
    "\n",
    "μ_τ, Σ_τ = get_traj_distribution(μ_w, Σ_w, des_duration)\n",
    "\n",
    "\n",
    "des_t = np.arange(0.0,des_duration,1/hz)\n",
    "μ_D = μ_τ.reshape((-1,des_t.shape[0]))\n",
    "\n",
    "uax.ax.plot(des_t,μ_D[0,:],'m',linewidth=5)\n",
    "uay.ax.plot(des_t,μ_D[1,:],'m',linewidth=5)\n",
    "\n",
    "σ_τ = np.sqrt(np.diag(Σ_τ))\n",
    "σ_D = σ_τ.reshape((-1,des_t.shape[0]))\n",
    "\n",
    "uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='m',alpha=0.3)\n",
    "uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='m',alpha=0.3)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "#implemented function, might still have to mess with matrices dimensions to work\n",
    "def conditioning(μ_w, Σ_w, y_t, Σ_y):\n",
    "    des_t = np.arange(0.0,des_duration,1/hz)\n",
    "    z = get_phase(des_t)\n",
    "    ϕ = radial_basis(z,c,h)\n",
    "    D = 2\n",
    "    Ψ = np.kron(np.eye(int(μ_w.shape[0]/B),dtype=int),ϕ)\n",
    "    \n",
    "    L = Σ_w.dot(Ψ).dot(inv(Σ_y + np.transpose(Ψ).dot(Σ_w).dot(Ψ)))\n",
    "    new_μ_w = μ_w + L.dot(y_t - np.transpose(Ψ).dot(μ_w))\n",
    "    new_Σ_w = Σ_w - Lnp.dot(np.transpose(Ψ)).dot(Σ_w)\n",
    "    \n",
    "    return new_μ_w, new_Σ_w\n",
    "\n",
    "#call to function\n",
    "conditioned_μ_w, conditioned_Σ_w = conditioning(μ_w, Σ_w, )\n",
    "\n",
    "#final steps are repeated again to create trajectory and plot it on graph\n",
    "μ_τ, Σ_τ = get_traj_distribution(conditioned_μ_w, conditioned_Σ_w, des_duration)\n",
    "\n",
    "\n",
    "des_t = np.arange(0.0,des_duration,1/hz)\n",
    "μ_D = μ_τ.reshape((-1,des_t.shape[0]))\n",
    "\n",
    "uax.ax.plot(des_t,μ_D[0,:],'m',linewidth=5)\n",
    "uay.ax.plot(des_t,μ_D[1,:],'m',linewidth=5)\n",
    "\n",
    "σ_τ = np.sqrt(np.diag(Σ_τ))\n",
    "σ_D = σ_τ.reshape((-1,des_t.shape[0]))\n",
    "\n",
    "uax.ax.fill_between(des_t, μ_D[0,:]-2*σ_D[0,:], μ_D[0,:]+2*σ_D[0,:], color='m',alpha=0.3)\n",
    "uay.ax.fill_between(des_t, μ_D[1,:]-2*σ_D[1,:], μ_D[1,:]+2*σ_D[1,:], color='m',alpha=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
