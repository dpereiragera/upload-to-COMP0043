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

def ou_calibrate_ml(X,dt):

    n=len(X)
    Sx=np.sum(X[:-1])
    Sy=np.sum(X[1:])
    Sxx=np.sum(np.array(X[:-1])**2)
    Sxy=np.sum(np.array(X[:-1])*np.array(X[1:]))
    Syy=np.sum(np.array(X[1:])**2)

    mu = (Sy*Sxx-Sx*Sxy)/(n*(Sxx-Sxy)-Sx**2+Sx*Sy);
    a = (Sxy-mu*Sx-mu*Sy+n*mu**2)/(Sxx-2*mu*Sx+n*mu**2);
    sigmah2 = (Syy-2*a*Sxy+a**2*Sxx-2*mu*(1-a)*(Sy-a*Sx)+n*mu**2*(1-a)**2)/n;
    alpha = -np.log(a)/dt;
    sigma = np.sqrt(sigmah2*2*alpha/(1-a**2))
    
    return mu,sigma,alpha