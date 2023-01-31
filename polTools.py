# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:14:24 2018

@author: Andrew
"""


"""
Created on Wed Feb 07 13:50:23 2018

@author: Andrew
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.table as Table
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_context("poster")
sns.set_style("ticks")

def calculate_P(Q,U,name="P"):
    return Table.Column(np.sqrt(Q**2 + U**2), name=name)

def phase_wrap(a):
    return np.concatenate((a-1,a,a+1))

def wrap(a):
    return np.concatenate((a,a,a))

def pa_angle(x, a, b):
    return a*x + b
    
def calculate_PA(Q, U, name="PA"):
    
    PA = np.rad2deg(0.5*np.arctan2(U, Q))
    i=0
    for pa in PA:
        if pa < 0:
            PA[i] = pa + 180
        i+=1
    
    return Table.Column(PA, name=name)   

def calculate_PA_error(q, u, sig_q, sig_u):
    p = np.sqrt(q**2 + u **2)
    return (1 / p**2) * np.sqrt((q * sig_q)**2 + (u * sig_u)**2)
    
def calculate_P_error(q, u, sig_q, sig_u):
    p = np.sqrt(q**2 + u **2)
    return (1 / p) * np.sqrt((q * sig_q)**2 + (u * sig_u)**2)
    
def rotate_pa(PA, Q, U, sig_q, sig_u, adjust=0):
    p = np.sqrt(Q**2 + U **2)
    sig_pa = (1 / p) * np.sqrt((Q * sig_q)**2 + (U * sig_u)**2)
    mean_pa = np.average(PA + adjust, weights = 1./sig_pa**2)
    
    print("Mean PA: ", mean_pa)
    q_rot = np.sqrt(Q**2 + U**2)*np.cos(2*np.deg2rad(PA - mean_pa))
    u_rot = np.sqrt(Q**2 + U**2)*np.sin(2*np.deg2rad(PA - mean_pa))

    return q_rot, u_rot

def rotate_pa_to_angle(Q, U, mean_pa):
    
    PA = np.rad2deg(0.5*np.arctan2(U, Q))
    
    print("Mean PA: ", mean_pa)
    q_rot = np.sqrt(Q**2 + U**2)*np.cos(2*np.deg2rad(PA - mean_pa))
    u_rot = np.sqrt(Q**2 + U**2)*np.sin(2*np.deg2rad(PA - mean_pa))

    return q_rot, u_rot
    
def plotting_BVR(phase, B, BE, V, VE, R, RE, newpoints, title=False, yaxislabel="%P", PA=False):
    fig, ax1 = plt.subplots(figsize=(12,6))

    phase_wrapped_v = phase_wrap(phase)
    pol_wrapped_b = wrap(B)
    yerr_wrapped_b = wrap(BE)
    pol_wrapped_v = wrap(V)
    yerr_wrapped_v = wrap(VE)
    pol_wrapped_r = wrap(R)
    yerr_wrapped_r = wrap(RE)
    
    
    ax1.errorbar(phase_wrapped_v, pol_wrapped_b, yerr = yerr_wrapped_b, fmt = "bo-", label="B")
    if newpoints:
        ax1.errorbar(phase_wrapped_v[newpoints], pol_wrapped_b[newpoints], yerr = yerr_wrapped_b[newpoints], fmt = "b*", markersize=30)
    
    ax1.errorbar(phase_wrapped_v, pol_wrapped_v, yerr = yerr_wrapped_v, fmt = "ko-", label="V")
    if newpoints:
        ax1.errorbar(phase_wrapped_v[newpoints], pol_wrapped_v[newpoints], yerr = yerr_wrapped_v[newpoints], fmt = "k*", markersize=30)
    
    ax1.errorbar(phase_wrapped_v, pol_wrapped_r, yerr = yerr_wrapped_r, fmt = "ro-", label="R")
    if newpoints:
        ax1.errorbar(phase_wrapped_v[newpoints], pol_wrapped_r[newpoints], yerr = yerr_wrapped_r[newpoints], fmt = "r*", markersize=30)
    
    ax1.set_xlabel("Phase")
    ax1.set_ylabel(yaxislabel)
    if not PA:
        plt.ylim((np.round(np.min(V)-0.75), np.round(np.max(V)+0.75)))
        ax1.vlines([0,1], np.min(V)-2, np.max(V)+2, colors="gray", linestyles = "dashed")
    else:
        if np.min(V) > 90:
            plt.ylim(90, 150)
        elif np.max(V) < 90:
            plt.ylim(0, 90)
        else:
            plt.ylim(0, 180)
        ax1.vlines([0,1], 0, 180, colors="gray", linestyles = "dashed")
    ax1.set_xlim((-0.2, 1.2))    
    ax1.legend(loc = "best")
    ax1.tick_params(direction="in", which='both')
    ax1.minorticks_on()
    #fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    if title:
        ax1.set_title(title)
        
def fix_negative_PA(PA):
    for i, pa in enumerate(PA):
        if pa < 0:
            PA[i] += 180
        
    return PA