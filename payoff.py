import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import lognorm
import QuantLib as ql
import time as time
from pytictoc import TicToc
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from QuantLib import *
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftshift, ifftshift

#Compute the scale the payoff and its Fourier transform
def payoff(x,xi,alpha,K,L,U,C,theta):

    # Scale
    S=C*np.exp(x)
    
    #Analytical Fourier transform of the payoff
    l=np.log(L/C) #lower log barrier
    k=np.log(K/C) #log strike
    u=np.log(U/C) #upper log barrier

    #Integration bounds
    if theta==1: #call

        #Payoff; see e.g. Green, Fusai, Abrahams 2010, Eq. (3.24)
        net_pay=theta*(S-K)
        net=len(net_pay)-len(net_pay[net_pay>0])
        g=np.concatenate((np.zeros(net),net_pay[net_pay>0]))*(S>=L)*(S<=U)*np.exp(alpha*x) #call
        a= max(l,k)
        b=u
    else: #put
        net=theta*(S-K)
        net[net<0]=0
        g=net*(S>=L)*(S<=U)*np.exp(alpha*x) #put
        a=min(k,u)
        b=l

    #Green, Fusai, Abrahams 2010 Eq. (3.26) with extension to put option
    xi2 =alpha+1j*xi
    G=C*((np.exp(b*(1+xi2))-np.exp(a*(1+xi2)))/(1+xi2)-(np.exp(k+b*xi2)-np.exp(k+a*xi2))/xi2) 

    #Eliminable discountinuities for xi=0, otherwise 0/0 = NaN:
    if alpha==0:
        G[int(np.floor(len(G)/2))]= C*(np.exp(b)-np.exp(a)-np.exp(k)*(b-a))
    elif alpha==-1:
        G[int(np.floor(len(G)/2))]= C*(b-a+np.exp(k-b)-np.exp(k-a))
    else:
        pass

    # Plot to compare the analytical and numerical payoffs
    # gn= fftshift(fft(ifftshift(G)))/((x[2]-x[1])*len(x))
    # plt.figure(1)
    # plt.plot(x,g,'g',x,np.real(gn),'r')
    # plt.xlabel('x')
    # plt.ylabel('g')
    # plt.legend(['analytical','numerical'])
    # if theta == 1:
    #     plt.title('Damped payoff function for a call option')
    # else:
    #     plt.title('Damped payoff function for a put option')
    # plt.show()

    return S,g,G