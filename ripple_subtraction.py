#!/usr/bin/env python
# coding: utf-8

# Imports

# In[ ]:


import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.io.fits as fits
import polTools
import glob
import lmfit
import xarray as xr


# Seaborn

# In[ ]:


sns.set()
sns.set_style("white")
sns.set_context("paper", font_scale=1.0)
sns.set_style("ticks")

filePattern = '/*.fits'


# Function defs

# In[ ]:


def viewstokes(stokes_Sw,err2_Sw,ok_w=[True],tcenter=0.):
    """Compute normalized stokes parameters, converts Q-U to P-T, for viewing
    Parameters
    ----------
    stokes_Sw: 2d float nparray(stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength
    var_Sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw
    ok_w: 1d boolean nparray(stokes,wavelength bin) 
       marking good stokes values. default all ok.
    Output: normalized stokes parameters and errors, linear stokes converted to pol %, PA
       Ignore covariance.  Assume binned first if necessary.
    """
    stokess,wavs = stokes_Sw.shape
    stokes_vw = np.zeros((stokess-1,wavs))
    err_vw = np.zeros((stokess-1,wavs))
    if (len(ok_w) == 1): ok_w = np.ones(wavs,dtype=bool)

    stokes_vw[:,ok_w] = 100.*stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]               # in percent
    err_vw[:,ok_w] = 100.*np.sqrt(err2_Sw[1:stokess,ok_w])/stokes_Sw[0,ok_w]     # error bar ignores covariance

    if (stokess >2):
        stokesP_w = np.zeros((wavs))
        stokesT_w = np.zeros((wavs))
        varP_w = np.zeros((wavs))
        varT_w = np.zeros((wavs))
        varpe_dw = np.zeros((2,wavs))
        varpt_w = np.zeros((wavs))
        # unnormalized linear polarization
        stokesP_w[ok_w] = np.sqrt(stokes_Sw[1,ok_w]**2 + stokes_Sw[2,ok_w]**2)  
        # PA in radians        
        stokesT_w[ok_w] = (0.5*np.arctan2(stokes_Sw[2,ok_w],stokes_Sw[1,ok_w]))
        # optimal PA folding  
        stokesT_w[ok_w] = (stokesT_w[ok_w]-(tcenter+np.pi/2.)+np.pi) % np.pi + (tcenter-np.pi/2.)                 
     # variance matrix eigenvalues, ellipse orientation
        varpe_dw[:,ok_w] = 0.5*(err2_Sw[1,ok_w]+err2_Sw[2,ok_w]                                      + np.array([1,-1])[:,None]*np.sqrt((err2_Sw[1,ok_w]-err2_Sw[2,ok_w])**2 + 4*err2_Sw[-1,ok_w]**2))
        varpt_w[ok_w] = 0.5*np.arctan2(2.*err2_Sw[-1,ok_w],err2_Sw[1,ok_w]-err2_Sw[2,ok_w])
     # linear polarization variance along p, PA   
        varP_w[ok_w] = varpe_dw[0,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2                  + varpe_dw[1,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2
        varT_w[ok_w] = varpe_dw[0,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2                  + varpe_dw[1,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2

        stokes_vw[0,ok_w] = 100*stokesP_w[ok_w]/stokes_Sw[0,ok_w]                  # normalized % linear polarization
        err_vw[0,ok_w] =  100*np.sqrt(err2_Sw[1,ok_w])/stokes_Sw[0,ok_w]
        stokes_vw[1,ok_w] = np.degrees(stokesT_w[ok_w])                            # PA in degrees
        err_vw[1,ok_w] =  0.5*np.degrees(np.sqrt(err2_Sw[2,ok_w])/stokesP_w[ok_w])

    return stokes_vw,err_vw

def binDataAngstrom(wave, stokes, goodData, error, binSize=10):
    binWavelength = (wave / binSize-0.5).astype(int) - int((wave / binSize-0.5).min())
    Bins = binWavelength.max()
    binWavelength[~goodData[1]] = -1
    
    stokesShape, empty = stokes.shape
    
    binArray = np.arange(Bins)
    binArrayOk = (binArray[:,None] == binWavelength[None,:])
    stokesBinned = (stokes[:,None,:] * binArrayOk).sum(axis=2)
    errorBinned = ((error[:stokesShape,None,:] + 2. * covar[:,None,:]) * binArrayOk).sum(axis=2)  
    wavelengthBinned = (wave[None,:] * binArrayOk).sum(axis=1) / binArrayOk.sum(axis=1)
    
    return stokesBinned, errorBinned, wavelengthBinned

def binDataError(wave, stokes, goodData, error, covar, bpm_Sw, binError=0.01):
    allowedgap = 5
    
    wgap0_g = np.where((bpm_Sw[0,:-1]==0) & (bpm_Sw[0,1:]!=0))[0] + 1
    wgap1_g = np.where((bpm_Sw[0,wgap0_g[0]:-1]!=0) & (bpm_Sw[0,wgap0_g[0]+1:]==0))[0]         +  wgap0_g[0] + 1
    wgap0_g = wgap0_g[0:wgap1_g.shape[0]]
    
    isbad_g = ((wgap1_g - wgap0_g) > allowedgap)
    
    stokes_sw, err_sw = viewstokes(stokes, error, ok_w=goodData,tcenter=np.pi/2)
    
    binvar_w = err_sw[0]**2
    bincovar_w = np.zeros_like(binvar_w)
    bincovar_w[goodData] = binvar_w[goodData]*covar[1,goodData]/error[1,goodData]
   
    ww = -1
    b = 0
    binWavelength = -1*np.ones((wavs))
    
    while (bpm_Sw[0,ww+1:]==0).sum() > 0:
        
        w = ww+1+np.where(bpm_Sw[0,ww+1:]==0)[0][0]
        
        cumsvar_W = np.cumsum((binvar_w[w:]+2.*bincovar_w[w:])*(bpm_Sw[0,w:]==0))                        /np.cumsum((bpm_Sw[0,w:]==0))**2
        
        err_W = np.sqrt(cumsvar_W)
        ww = wavs                                       # stopping point override: end
        
        nextbadgap = np.where(isbad_g & (wgap0_g > w))[0]
        
        if nextbadgap.size: ww = wgap0_g[nextbadgap[0]] - 1   # stopping point override: before bad gap
        
        dw = np.where(err_W[:ww-w] < binError)[0]
        
        if dw.size: ww = w + dw[0]                      # err goal is reached first
            
        binWavelength[w:ww+1] = b
        b += 1
        
    binWavelength[bpm_Sw[0]>0] = -1
    Bins  = b
    
    stokesShape, empty = stokes.shape
    
    binArray = np.arange(Bins)
    binArrayOk = (binArray[:,None] == binWavelength[None,:])
    stokesBinned = (stokes[:,None,:]*binArrayOk).sum(axis=2)
    errorBinned = ((error[:stokesShape,None,:] + 2.*covar[:,None,:])*binArrayOk).sum(axis=2)  
    wavelengthBinned = (wave[None,:]*binArrayOk).sum(axis=1)/binArrayOk.sum(axis=1)
    
    return stokesBinned, errorBinned, wavelengthBinned


# In[ ]:


def fileLoad(folder):
    '''Loads a fits file'''
    dataFile = glob.glob(folder+filePattern)

    #Open fits file
    with fits.open(dataFile[0]) as hdul:

       # stokes I, Q, U values in [0, :], [1, :], [2, :]
        stokesSw = hdul['SCI'].data[:,0,:]
        #stokes errors (I, Q, U)
        varSw = hdul['VAR'].data[:,0,:]
        covarSw = hdul['COV'].data[:,0,:]
        deltaWave = float(hdul['SCI'].header['CDELT1'])
        #get starting wavelength
        wave0 = float(hdul['SCI'].header['CRVAL1'])
        #get wavelength axis size
        waves = int(hdul['SCI'].header['NAXIS1'])

        bpm_Sw = hdul['BPM'].data[:,0,:]
        ok_Sw = (bpm_Sw==0)

        wavelengths = wave0 + deltaWave*np.arange(waves)

        return stokesSw, varSw, covarSw, wavelengths, ok_Sw, bpm_Sw
    
def fileWrite(folder, stokesSw):
    '''Loads a fits file'''
    dataFile = glob.glob(folder+filePattern)

    #Open fits file
    with fits.open(dataFile[0]) as hdul:

       # stokes I, Q, U values in [0, :], [1, :], [2, :]
        hdul['SCI'].data[:,0,:] = stokesSw

        hdul.writeto(dataFile[0]+"_ripple_removed")

        return

def readData(fileName):
    data = []
    dataError = []
    dataCovar = []
    dataWave = []
    dataOk = []
    dataBPM = []

    #folder = "F:/Andrew's Dropbox/Dropbox/"
    folder = ""
    fileNameTotal = folder + fileName
    
    folderList = glob.glob(fileNameTotal)
    
    for folder in folderList:
        print(folder)
        stokes, stokesError, stokesCovar, waves, ok, bpm = fileLoad(folder)
        data.append(stokes)
        dataError.append(stokesError)
        dataCovar.append(stokesCovar)
        dataWave.append(waves)
        dataOk.append(ok)
        dataBPM.append(bpm)
  
    return data, dataError, dataCovar, dataWave, dataOk, dataBPM

def writeData(fileName, data):
    #folder = "F:/Andrew's Dropbox/Dropbox/"
    folder = ""
    fileNameTotal = folder + fileName
    
    folderList = glob.glob(fileNameTotal)
    
    for (folder, stokes) in zip(folderList, data):
        print(folder)
        fileWrite(folder, stokes)
  
    return


# In[ ]:


data2017, dataError2017, dataCovar2017, dataWave2017, dataOk2017, dataBPM2017 = readData("D:/WR_Data/WR042/20*")
#data2018, dataError2018, dataCovar2018, dataWave2018, dataOk2018, dataBPM2018 = readData("/RSS 2018-1/observations/WR113/20*")


# In[ ]:


fig, ax = plt.subplots(figsize = (6,4), dpi=150)

ripple = ascii.read("PA_Ripple.txt", data_start=2)

ripple_interp = np.polynomial.polynomial.polyfit(ripple["wavl"][(ripple["wavl"] > 4100) & (ripple["wavl"]<7800)], ripple["dPA"][(ripple["wavl"] > 4100) & (ripple["wavl"]<7800)], deg=15)
ripple_interp2 = sp.interpolate.interp1d(ripple["wavl"], ripple["dPA"], kind='cubic', fill_value='extrapolate')

wavs = np.linspace(4000, 10000, 1000)
#pp = sp.interpolate.PPoly.from_spline(ripple_interp2)

ax.plot(ripple["wavl"], ripple["dPA"], "ko")
ax.plot(wavs, ripple_interp2(wavs))
#ax.plot(wavs, np.polyval(ripple_interp[::-1], wavs))
#ax.plot(wavs, sp.interpolate.splev(wavs, ripple_interp2))
#ax.plot([4300, 4600, 4850, 5200, 5600, 6150, 6650, 7000, 8600, 9180], np.ones(10), "rx")
ax.set_xlim(3500, 10500)
ax.set_ylim(-2, 2)
ax.set_xlabel("$\lambda~(\AA)$")
ax.set_ylabel("Position angle ($\degree$)")


fig, ax = plt.subplots(figsize = (6,4), dpi=150)

ripple = ascii.read("PA_Ripple.txt", data_start=2)

q_ripple = 2*np.cos(2 * np.deg2rad(ripple["dPA"] + 22.5))
u_ripple = 2*np.sin(2 * np.deg2rad(ripple["dPA"] + 22.5))

ripple_q_interp = sp.interpolate.interp1d(ripple["wavl"], q_ripple, kind='cubic', fill_value='extrapolate')
ripple_u_interp = sp.interpolate.interp1d(ripple["wavl"], u_ripple, kind='cubic', fill_value='extrapolate')

ax.plot(ripple["wavl"], q_ripple, "ko")
ax.plot(ripple["wavl"], u_ripple, "ko")
ax.plot(wavs, ripple_q_interp(wavs))
ax.plot(wavs, ripple_u_interp(wavs))
ax.set_xlim(3500, 10500)
#ax.set_ylim(-2, 2)

fig, ax = plt.subplots(figsize = (6,6), dpi=150)
ax.plot(ripple_q_interp(wavs), ripple_u_interp(wavs))

# In[ ]:

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def fitting_poly(params, x, q=None, qerr=None, u=None, uerr=None, size=309):
    
    model_q = np.zeros((len(q), size))
    model_u = np.zeros((len(u), size))
    i=0
    
    for i in range(len(q)):
        model_q[i] = params["A_%i"%(i+1)] * np.cos(2 * np.deg2rad(ripple_interp2(x[i] + params["x"]) + params["rot_%i"%(i+1)])) \
        + params["q00_%i"%(i+1)] + params["P_max"] * np.cos(2 * (params["theta0"] + params["k"] * 1./(x[i]*1e-4))) \
         * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x[i]*1e-4))**2)

        model_u[i] = params["A_%i"%(i+1)] * np.sin(2 * np.deg2rad(ripple_interp2(x[i] + params["x"]) + params["rot_%i"%(i+1)])) \
        + params["u00_%i"%(i+1)] + params["P_max"] * np.sin(2 * (params["theta0"] + params["k"] * 1./(x[i]*1e-4))) \
         * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x[i]*1e-4))**2)

    if q is None:
        return model_q, model_u
    
    residual_q = np.zeros((len(q), size))
    residual_u = np.zeros((len(u), size))
    
    weighted_q = np.zeros((len(q), size))
    weighted_u = np.zeros((len(u), size))
    
    for i, date in enumerate(q):
        residual_q[i] = date - model_q[i]
        weighted_q[i] = np.sqrt(residual_q[i]**2/qerr[i]**2)
    
    for i, date in enumerate(u):
        residual_u[i] = date - model_u[i]
        weighted_u[i] = np.sqrt(residual_u[i]**2/uerr[i]**2) 
           
    return np.concatenate((weighted_q.flatten(), weighted_u.flatten()))

def fitting_poly_no_isp(params, x, q=None, qerr=None, u=None, uerr=None, size=309):
    
    model_q = np.zeros((len(q), size))
    model_u = np.zeros((len(u), size))
    i=0
    
    for i in range(len(q)):
        model_q[i] = np.cos(2 * (params["A_%i"%(i+1)] * np.deg2rad(ripple_interp2(x[i] * params["x_%i"%(i+1)])) + params["rot_%i"%(i+1)])) \
        + params["q00_%i"%(i+1)]

        model_u[i] = np.sin(2 * (params["A_%i"%(i+1)] * np.deg2rad(ripple_interp2(x[i] * params["x_%i"%(i+1)])) + params["rot_%i"%(i+1)])) \
        + params["u00_%i"%(i+1)]

    if q is None:
        return model_q, model_u
    
    residual_q = np.zeros((len(q), size))
    residual_u = np.zeros((len(u), size))
    
    weighted_q = np.zeros((len(q), size))
    weighted_u = np.zeros((len(u), size))
    
    for i, date in enumerate(q):
        residual_q[i] = date - model_q[i]
        weighted_q[i] = np.sqrt(residual_q[i]**2/qerr[i]**2)
    
    for i, date in enumerate(u):
        residual_u[i] = date - model_u[i]
        weighted_u[i] = np.sqrt(residual_u[i]**2/uerr[i]**2) 
           
    return np.concatenate((weighted_q.flatten(), weighted_u.flatten()))

def fitting_isp(params, x, q=None, qerr=None, u=None, uerr=None, size= 309):
    
    model_q = np.zeros((len(q), size))
    model_u = np.zeros((len(u), size))
    i=0
    
    for i in range(len(q)):
        model_q[i] = params["q00_%i"%(i+1)] + params["P_max"] * np.cos(2 * (params["theta0"] + params["k"] * 1./(x[i]*1e-4)))         * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x[i]*1e-4))**2)

        model_u[i] = params["u00_%i"%(i+1)] + params["P_max"] * np.sin(2 * (params["theta0"] + params["k"] * 1./(x[i]*1e-4)))         * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x[i]*1e-4))**2)

    if q is None:
        return model_q, model_u
    
    residual_q = np.zeros((len(q), size))
    residual_u = np.zeros((len(u), size))
    
    weighted_q = np.zeros((len(q), size))
    weighted_u = np.zeros((len(u), size))
    
    for i, date in enumerate(q):
        residual_q[i] = date - model_q[i]
        weighted_q[i] = residual_q[i]/qerr[i]
    
    for i, date in enumerate(u):
        residual_u[i] = date - model_u[i]
        weighted_u[i] = residual_u[i]/uerr[i]
        
    return np.concatenate((weighted_q.flatten(), weighted_u.flatten()))

def fit_position_angle(params, x, pa=None, err=None, size= 309):

    model = np.zeros((len(pa), size))
    i=0
    
    for i in range(len(pa)):
        model[i] = params["A_%i"%(i+1)] * np.deg2rad(ripple_interp2(x[i] + params["x_%i"%(i+1)])) + params["rot_%i"%(i+1)]

    if q is None:
        return model
    
    residual = np.zeros((len(pa), size))
    
    weighted = np.zeros((len(pa), size))
    
    for i, date in enumerate(pa):
        residual[i] = date - model[i]
        weighted[i] = residual[i]/err[i]
        
    return weighted.flatten()

def fitting_poly_q_i(params, x, i):
    
    model_q = np.cos(2 * (np.deg2rad(ripple_interp2(x + params["x"]) + params["rot_%i"%(i+1)])))     + params["q00_%i"%(i+1)] + params["P_max"] * np.cos(2 * (params["theta0"] + params["k"] * 1./(x*1e-4)))     * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x*1e-4))**2)

    return model_q

def fitting_position_angle_out(params, x, i):
    
    model = params["A_%i"%(i+1)] * np.deg2rad(ripple_interp2(x + params["x_%i"%(i+1)])) + params["rot_%i"%(i+1)]

    return model
 
def fitting_poly_u_i(params, x, i):
    
    model_u = np.sin(2 * (np.deg2rad(ripple_interp2(x + params["x"]) + params["rot_%i"%(i+1)])))     + params["u00_%i"%(i+1)] + params["P_max"] * np.sin(2 * (params["theta0"] + params["k"] * 1./(x*1e-4)))     * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x*1e-4))**2)

    return model_u

def fitting_q_i(params, x, i):
    
    model_q = np.cos(2 * (params["A_%i"%(i+1)] * np.deg2rad(ripple_interp2(x + params["x_%i"%(i+1)])) + params["rot_%i"%(i+1)]))\
               #+ params["q00_%i"%(i+1)]

    return model_q

def fitting_u_i(params, x, i):
    
    model_u = np.sin(2 * (params["A_%i"%(i+1)] * np.deg2rad(ripple_interp2(x + params["x_%i"%(i+1)])) + params["rot_%i"%(i+1)]))\
               #+ params["u00_%i"%(i+1)]

    return model_u

def fitting_q_isp_i(params, x, i):
    
    model_q = params["P_max"] * np.cos(2 * (params["theta0"] + params["k"] * 1./(x*1e-4)))     * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x*1e-4))**2)

    return model_q
 
def fitting_u_isp_i(params, x, i):
    
    model_u = params["P_max"] * np.sin(2 * (params["theta0"] + params["k"] * 1./(x*1e-4)))     * np.exp(-1.7 * params["lambda_max"] * np.log(params["lambda_max"] * 1./(x*1e-4))**2)

    return model_u

min_waves_table = []
max_waves_table = []

for wave in dataWave2017:
    min_waves_table.append(np.min(wave))
    max_waves_table.append(np.max(wave))
    
min_wave = np.min(min_waves_table)
max_wave = np.max(max_waves_table)
    
qu_table = []

#WR42 ISP
par = lmfit.Parameters()
par.add('P_max', value = 1.177)
par.add('theta0', value = np.deg2rad(-46.3), min=-np.pi, max=np.pi)
par.add('k', value= np.deg2rad(0), vary=False)
par.add('lambda_max', value = 0.568)

#WR79 ISP
#par = lmfit.Parameters()
#par.add('P_max', value = 0.376)
#par.add('theta0', value = np.deg2rad(-81.5), min=-np.pi, max=np.pi)
#par.add('k', value= np.deg2rad(4.7), vary=False)
#par.add('lambda_max', value = 0.595)

i=0
for (stokes, error, covar, wave, goodData) in zip(data2017, dataError2017, dataCovar2017, dataWave2017, dataOk2017):
    
    if np.min(wave) > min_wave or np.max(wave) < max_wave:
        padded_wave = np.pad(wave, (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='linear_ramp', end_values=(min_wave, max_wave))

        goodData = np.array((np.pad(goodData[0], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                             np.pad(goodData[1], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                             np.pad(goodData[2], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge')))
        
        stokes = np.array((np.pad(stokes[0], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                           np.pad(stokes[1], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                           np.pad(stokes[2], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge')))
        
        error = np.array((np.pad(error[0], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                          np.pad(error[1], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                          np.pad(error[2], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge')))
        
        covar = np.array((np.pad(covar[0], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                          np.pad(covar[1], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge'), \
                          np.pad(covar[2], (int(np.min(wave) - min_wave), int(max_wave - np.max(wave))), mode='edge')))

    binSize = 10
    stokesBinned, errorBinned, wavelengthBinned = binDataAngstrom(padded_wave, stokes, goodData, error, binSize=binSize)

    q, u = (stokesBinned[1]/stokesBinned[0])*100, (stokesBinned[2]/stokesBinned[0])*100
    
    q_err = (np.sqrt(errorBinned[1])/stokesBinned[0])*100
    u_err = (np.sqrt(errorBinned[2])/stokesBinned[0])*100
    
    q_isp = fitting_q_isp_i(par, wavelengthBinned, i)
    u_isp = fitting_u_isp_i(par, wavelengthBinned, i)
    
    q -= q_isp
    u -= u_isp
    
    mask1 = np.where((wavelengthBinned > 4570) & (wavelengthBinned < 4760))
    mask2 = np.where((wavelengthBinned > 5620) & (wavelengthBinned < 5900))
    mask3 = np.where((wavelengthBinned > 6525) & (wavelengthBinned < 6615))
    mask4 = np.where((wavelengthBinned > 6700) & (wavelengthBinned < 6800))
    mask5 = np.where((wavelengthBinned > 6990) & (wavelengthBinned < 7100))
    mask6 = np.where((wavelengthBinned > 7200) & (wavelengthBinned < 7260))
    
    q[mask1] = np.nan
    q[mask2] = np.nan
    q[mask3] = np.nan
    q[mask4] = np.nan
    q[mask5] = np.nan
    q[mask6] = np.nan
    
    u[mask1] = np.nan
    u[mask2] = np.nan
    u[mask3] = np.nan
    u[mask4] = np.nan
    u[mask5] = np.nan
    u[mask6] = np.nan
    
    pa = np.deg2rad(polTools.calculate_PA(q, u))
    pa_err = np.deg2rad(polTools.calculate_PA_error(q, u, q_err, u_err))
    
    #for WR79
    #if i == 0:
    #    for j, PA in enumerate(pa):
    #        if PA < np.pi/2:
    #            pa[j] += np.pi
    
    nans, x_nans= nan_helper(wavelengthBinned)
    wavelengthBinned[nans]= np.interp(x_nans(nans), x_nans(~nans), wavelengthBinned[~nans])
    
    qu_da = np.array((wavelengthBinned, q, q_err, u, u_err, pa, pa_err))
    
    qu_table.append(qu_da)
    
    i+=1

size = len(wavelengthBinned)

qu_table = np.stack(qu_table)
    
qu_da2 = xr.DataArray(qu_table, dims=['date', 'data', 'rows'], coords={'data': ['wavelength', 'q', 'qerr', 'u', 'uerr', 'PA', 'PAerr']})

par = lmfit.Parameters()
for i in range(len(qu_da2.loc[:, 'q'])):    
    par.add('A_%i'%(i+1), value = 0.0)
    #par.add('P_%i'%(i+1), value = 1.0)
    par.add('rot_%i'%(i+1), value = np.mean(qu_da2.loc[i, 'PA']), min=0, max=np.pi)
    #par.add('xu_%i'%(i+1), value = 1.0)
    #par.add('q00_%i'%(i+1), value = np.mean(qu_da2.loc[i, 'q'])-1)
    #par.add('u00_%i'%(i+1), value = np.mean(qu_da2.loc[i, 'u'])+1)
    par.add('x_%i'%(i+1), value = 0.0)
    
#par.add('P_max', value = 1.177)
#par.add('theta0', value = np.deg2rad(-46.3), min=-np.pi, max=np.pi)
#par.add('k', value= 0.0, vary=False)
#par.add('lambda_max', value = 0.568)
    
#mini = lmfit.Minimizer(fitting_poly_no_isp, par, fcn_args=(qu_da2.loc[:, 'wavelength'],),\
#                       fcn_kws={'q':qu_da2.loc[:, 'q'], 'qerr':qu_da2.loc[:, 'qerr'], \
#                                'u':qu_da2.loc[:, 'u'], 'uerr':qu_da2.loc[:, 'uerr'], \
#                                'size':size}, nan_policy='omit')
    
mini = lmfit.Minimizer(fit_position_angle, par, fcn_args=(qu_da2.loc[:, 'wavelength'],),\
                       fcn_kws={'pa': qu_da2.loc[:, 'PA'], 'err': qu_da2.loc[:, 'PAerr'], \
                                'size':size}, nan_policy='omit')
 
out = mini.least_squares(loss='huber', f_scale = 1.345)
    
#out = mini.minimize(method='nelder')


# In[ ]:


print(lmfit.fit_report(out, show_correl=False))

wavs = np.linspace(4200, 7250, 1000)

fig, subplots = plt.subplots(20, 1, figsize = (5, 20), sharex=True, dpi=150)

fig.subplots_adjust(hspace=0.0)

for i in range(len(qu_da2.loc[:, 'PA'])): 
    ax = subplots.flat[i]
    ax.step(qu_da2.loc[i, 'wavelength'], qu_da2.loc[i, 'PA'], where='mid')
    ax.plot(wavs, fitting_position_angle_out(out.params, wavs, i))
    ax.set_xlim(4000, 7500)
    ax.set_ylabel('PA')
    
ax.set_xlabel('Wavelength')

fig, subplots = plt.subplots(20, 1, figsize = (5, 20), sharex=True, dpi=150)

fig.subplots_adjust(hspace=0.0)    
    
for i in range(len(qu_da2.loc[:, 'q'])):
    p = np.mean(np.sqrt(qu_da2.loc[i, 'u']**2 + qu_da2.loc[i, 'q']**2))
     
    ax = subplots.flat[i]
    ax.step(qu_da2.loc[i, 'wavelength'], qu_da2.loc[i, 'q'], where='mid')
    ax.plot(wavs, p.data * fitting_q_i(out.params, wavs, i))
    ax.set_xlim(4000, 7500)
    ax.set_ylabel('%q')
    
ax.set_xlabel('Wavelength')

fig.savefig("ripple_fit_q.eps")

fig, subplots = plt.subplots(20, 1, figsize = (5, 20), sharex=True, dpi=150)

fig.subplots_adjust(hspace=0.0)

for i in range(len(qu_da2.loc[:, 'u'])):
    p = np.mean(np.sqrt(qu_da2.loc[i, 'u']**2 + qu_da2.loc[i, 'q']**2))

    ax = subplots.flat[i]
    ax.step(qu_da2.loc[i, 'wavelength'], qu_da2.loc[i, 'u'], where='mid')
    ax.plot(wavs, p.data * fitting_u_i(out.params, wavs, i))
    ax.set_xlim(4000, 7500)
    ax.set_ylabel('%u')
    
ax.set_xlabel('Wavelength')

fig.savefig("ripple_fit_u.eps")

# In[ ]:

for i in range(len(qu_da2.loc[:, 'q'])):    
    out.params['rot_%i'%(i+1)].set(0.0)
    #out.params['q00_%i'%(i+1)].set(0.0)
    #out.params['u00_%i'%(i+1)].set(0.0)

fig, subplots = plt.subplots(20, 1, figsize = (5, 10), sharex=True, dpi=150)

fig.subplots_adjust(hspace=0.0)    
print('q')     
for i in range(len(qu_da2.loc[:, 'q'])):
    pa = qu_da2.loc[i, 'PA'] - fitting_position_angle_out(out.params, qu_da2.loc[i, 'wavelength'], i)
    p = np.sqrt(qu_da2.loc[i, 'u']**2 + qu_da2.loc[i, 'q']**2)

    ax = subplots.flat[i]
    #ax.step(qu_da2.loc[i, 'wavelength'], qu_da2.loc[i, 'q'] - fitting_q_i(out.params, qu_da2.loc[i, 'wavelength'], i), where='mid')
    ax.step(qu_da2.loc[i, 'wavelength'], p * np.cos(2 * pa), where='mid')
    ax.set_xlim(4000, 7500)
    ax.set_ylabel('%q')
    
    p = np.ma.array(p, mask=np.isnan(p))
    print(np.ptp(p * fitting_q_i(out.params, qu_da2.loc[i, 'wavelength'], i))/2)
    
    
ax.set_xlabel('Wavelength')

fig, subplots = plt.subplots(20, 1, figsize = (5, 10), sharex=True, dpi=150)

fig.subplots_adjust(hspace=0.0)
print('u')
for i in range(len(qu_da2.loc[:, 'u'])):
    pa = qu_da2.loc[i, 'PA'] - fitting_position_angle_out(out.params, qu_da2.loc[i, 'wavelength'], i)
    p = np.sqrt(qu_da2.loc[i, 'u']**2 + qu_da2.loc[i, 'q']**2)
    
    ax = subplots.flat[i]
    #ax.step(qu_da2.loc[i, 'wavelength'], qu_da2.loc[i, 'u'] - fitting_u_i(out.params, qu_da2.loc[i, 'wavelength'], i), where='mid')
    ax.step(qu_da2.loc[i, 'wavelength'], p * np.sin(2 * pa), where='mid')
    ax.set_xlim(4000, 7500)
    ax.set_ylabel('%u')
    
    p = np.ma.array(p, mask=np.isnan(p))
    print(np.ptp(p * fitting_u_i(out.params, qu_da2.loc[i, 'wavelength'], i))/2)
    q = p * fitting_q_i(out.params, qu_da2.loc[i, 'wavelength'], i)
    u = p * fitting_u_i(out.params, qu_da2.loc[i, 'wavelength'], i)
    X = np.ma.stack((q, u), axis=0)
    print(np.ma.cov(X))

ax.set_xlabel('Wavelength')

# In[ ]:

i=0

data_out = []

for (stokes, wave) in zip(data2017, dataWave2017):
    
    q, u = (stokes[1]/stokes[0])*100, (stokes[2]/stokes[0])*100
    
    nans, x_nans= nan_helper(wave)
    wave[nans]= np.interp(x_nans(nans), x_nans(~nans), wave[~nans])
        
    pa = (0.5 * np.arctan2(u, q)) - fitting_position_angle_out(out.params, wave, i)
    p = np.sqrt(q**2 + u**2)
    
    q = p * np.cos(2 * pa)
    u = p * np.sin(2 * pa)
    
    stokes[1] = q/100*stokes[0]
    stokes[2] = u/100*stokes[0]
    
    data_out.append(stokes)
    
    i+=1

writeData("D:/WR_Data/WR079/20*", data_out)


# In[ ]:




