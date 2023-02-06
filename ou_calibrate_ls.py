import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import lognorm
from pytictoc import TicToc
from scipy.stats import ncx2

def ou_calibrate_ls(X,dt):

    n=len(X)
    Sx=np.sum(X[:-1])
    Sy=np.sum(X[1:])
    Sxx=np.sum(np.array(X[:-1])**2)
    Sxy=np.sum(np.array(X[:-1])*np.array(X[1:]))
    Syy=np.sum(np.array(X[1:])**2)

    a=(n*Sxy-Sx*Sy)/(n*Sxx-Sx**2)
    b=(Sy-a*Sx)/n
    sd2 = (n*Syy-Sy**2-a*(n*Sxy-Sx*Sy))/(n*(n-2))

    mu = b/(1-a)
    alpha = -np.log(a)/dt
    sigma = np.sqrt(sd2*2*alpha/(1-a**2))
    
    return mu,sigma,alpha