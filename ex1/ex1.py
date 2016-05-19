import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import *


class ex1:

    def __init__(self):
        data=np.loadtxt("ex1data2.txt",delimiter=",")
        self.X=data[:,0:-1]
        self.y=data[:,-1]
        self.m=len(self.X)
        self.theta=np.zeros(self.X.shape[1]+1)
        self.iterations=1500
        self.alpha=0.1
        self.normX,self.mu,self.sigma=self.normFeature()
        self.newX=np.c_[np.ones(self.m),self.normX] 

    def normFeature(self):
        mu=np.mean(self.X,axis=0)
        sigma=np.std(self.X,axis=0)
        muArr=tile(mu, (self.X.shape[0],1)) 
        sigmaArr=tile(sigma, (self.X.shape[0],1)) 
        X=(self.X-muArr)/sigmaArr
        return(X, mu, sigma)

    '''
    def pltData(self):
        plt.plot(self.X,self.y,'ro')
        plt.plot(self.X,self.theta.dot(self.newX.T))
        plt.ylabel('Profit in $10,1000s')
        plt.xlabel('Population of City in 10,000s')
        plt.show()
    '''

    def calCostFun(self):
       val=1./(2.*self.m)*np.sum(((self.theta.dot(self.newX.T))-self.y)**2) 
       return(val)

    def gradDescent(self):
       valCost=[]
       thetaHis=self.theta
       for i in range(self.iterations):
           for j in range(len(self.theta)):
               self.theta[j]=self.theta[j]-self.alpha*1./self.m*np.sum(((self.theta.dot(self.newX.T))-self.y)*self.newX[:,j])
           valCost.append(self.calCostFun())
           thetaHis=np.concatenate((thetaHis,self.theta))
       thetaHis=thetaHis.reshape(self.iterations+1,len(self.theta))
       plt.plot(valCost)
       plt.show()
       return(self.theta,thetaHis)
    '''
    def visCostFun(self,thetaHis):
       fig = plt.figure()
       ax = fig.gca(projection='3d')

       theta0Rg=np.arange(-10, 10, 0.1)
       theta1Rg=np.arange(-1.,4., 0.01)
       valJ=np.zeros((len(theta0Rg),len(theta1Rg)))
       theta0Grid, theta1Grid=np.meshgrid(theta0Rg,theta1Rg)
       theta=np.zeros(2)
       for i in range(len(theta0Rg)):
           for j in range(len(theta1Rg)):
               theta[0]=theta0Rg[i]
               theta[1]=theta1Rg[j]
               valJ[i,j]=1./(2.*self.m)*np.sum(((theta.dot(self.newX.T))-self.y)**2) 
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
       plt.plot([theta0Rg[i]],[theta1Rg[j]],'-ro')
       for ele in thetaHis:
          plt.plot([ele[0]],[ele[1]],'b*')
       plt.xlabel("theta0")
       plt.ylabel("theta1")
       plt.show()
    '''

    def check(self):
        theta = np.linalg.lstsq(self.newX, self.y)[0]
        return(theta)
    
    def normEqu(self):
        newX=np.c_[np.ones(self.m),self.X]
        theta=np.linalg.inv((newX.T).dot(newX)).dot(newX.T).dot(self.y)
        return(theta)

OB=ex1()
OB.iterations=500

OB.alpha=0.1
theta,thetaHis=OB.gradDescent()

thetaCheck=OB.check()
thetaNorm=OB.normEqu()
print('---------------------------')
print("theta from Norm Equ:")
print(thetaNorm)
print('---------------------------')
print("theta from own code")
print(theta)
print('---------------------------')
print("theta from library")
print(thetaCheck)
print('---------------------------')


price = np.sum(np.array([1.,(1650.-OB.mu[0])/OB.sigma[0], (3.-OB.mu[1])/OB.sigma[1]])*theta)
print('                           ')
print('---------------------------')
print('---------------------------')
print("price calculated by my code")
print(price)
print('---------------------------')
price=np.sum(np.array([1,1650,3])*thetaNorm)
print("price calculated by norm equation")
print(price)
print('                           ')
