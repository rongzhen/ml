import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LogNorm
from scipy import *


class ex1:

    def __init__(self):
        data=np.loadtxt("ex1data1.txt",delimiter=",")
        self.X=data[:,0]
        self.y=data[:,1]
        self.m=len(self.X)
        self.theta=np.zeros(2)
        self.iterations=1500
        self.alpha=0.01
        self.newX=np.c_[np.ones(self.m),self.X] 
        

    def pltData(self):
        plt.plot(self.X,self.y,'ro')
        plt.plot(self.X,self.theta.dot(self.newX.T))
        plt.ylabel('Profit in $10,1000s')
        plt.xlabel('Population of City in 10,000s')
        plt.show()

    def calCostFun(self):
       y=self.y
       val=1./(2.*self.m)*np.sum(((self.theta.dot(self.newX.T))-y)**2) 
       return(val)

    def gradDescent(self):
       y=self.y
       valCost=[]
       thetaHis=self.theta
       for i in range(self.iterations):
           self.theta[0]=self.theta[0]-self.alpha*1./self.m*np.sum((self.theta.dot(self.newX.T))-y)
           self.theta[1]=self.theta[1]-self.alpha*1./self.m*np.sum(((self.theta.dot(self.newX.T))-y)*self.X)
           valCost.append(self.calCostFun())
           thetaHis=np.concatenate((thetaHis,self.theta))
       thetaHis=thetaHis.reshape(self.iterations+1,2)
       plt.plot(valCost)
       plt.show()
       return(self.theta,thetaHis)

    def visCostFun(self,thetaHis):
       fig = plt.figure()
       ax = fig.gca(projection='3d')

       theta0=np.arange(-10, 10, 0.1)
       theta1=np.arange(-1.,4., 0.01)
       valJ=np.zeros((len(theta0),len(theta1)))
       theta0Grid, theta1Grid=np.meshgrid(theta0,theta1)
       for i in range(len(theta0)):
           for j in range(len(theta1)):
               self.theta[0]=theta0[i]
               self.theta[1]=theta1[j]
               valJ[i,j]=self.calCostFun()
       i,j = np.unravel_index(valJ.argmin(), valJ.shape)
#       surf = ax.plot_surface(theta0Grid, theta1Grid, valJ.T, rstride=1, cstride=1, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
       surf = ax.plot_surface(theta0Grid, theta1Grid, valJ.T)
       plt.xlabel("theta0")
       plt.ylabel("theta1")
       plt.show()

       levels = np.arange(-30.,30.,5)
       CS=plt.contour(theta0Grid, theta1Grid, valJ.T, levels)
       plt.clabel(CS, inline=1, fontsize=10)
       plt.plot([theta0[i]],[theta1[j]],'-ro')
       for ele in thetaHis:
          plt.plot([ele[0]],[ele[1]],'b*')
       plt.xlabel("theta0")
       plt.ylabel("theta1")
       plt.show()

OB=ex1()
OB.iterations=5000
val=OB.calCostFun()
theta,thetaHis=OB.gradDescent()
print(theta)
OB.pltData()
OB.visCostFun(thetaHis)
