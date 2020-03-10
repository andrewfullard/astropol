# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:43:50 2019

@author: Andrew
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.io.fits as fits
from astropy.table import Table
from scipy.interpolate import interpolate
import polTools
import os
import glob

# To run: change line 24 as noted in comment below. Enter star name (same as folder name containing the dated folders). 
# Press Enter, then enter the filter name: johnson, stroemgren or narrow. Press Enter. Results are saved in a text file.


# Change this path so that it points to a folder named by the WR star that contains folders with dates in the format YYYYMMDD. 
# The folders should then contain .fits files of the data to reduce
folderList = glob.glob("C:/")

filePattern = '/*.fits'

# Load fits file
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

# Read fits file
def readData(fileName):
    data = []
    dataError = []
    dataCovar = []
    dataWave = []
    dataOk = []
    dataBPM = []
    
    folderList = glob.glob(fileName)
    
    for folder in folderList:
        print(folder[-8:])
        
        data.append(stokes)
        dataError.append(stokesError)
        dataCovar.append(stokesCovar)
        dataWave.append(waves)
        dataOk.append(ok)
        dataBPM.append(bpm)
  
    return data, dataError, dataCovar, dataWave, dataOk, dataBPM

# Location of the filter files
filter_Path = "Filters/"

# Load filters
filter_B_Standard = ascii.read(filter_Path + "B Filter Standard.txt")
filter_V_Standard = ascii.read(filter_Path + "V Filter Standard.txt")
filter_R_Standard = ascii.read(filter_Path + "R Filter Standard.txt")
filter_B_Stroemgren = ascii.read(filter_Path + "Stroemgren_b.txt")
filter_B_WR = ascii.read(filter_Path + "B Filter WR.txt")
filter_V_WR = ascii.read(filter_Path + "V Filter WR.txt")
filter_R_WR = ascii.read(filter_Path + "R Filter WR.txt")

# Sort filters
filter_B_Standard.sort('Wavelength')
filter_V_Standard.sort('Wavelength')
filter_R_Standard.sort('Wavelength')
filter_B_Stroemgren.sort('Wavelength')

# User entry
datapath = input("Enter the name of the star: ")
Star_type = input("Enter filter type: ")

# Output table format
final_result = Table(names=("Date", "BQ", "BU", "BQerr", "BUerr", "VQ", "VU", "VQerr", "VUerr", "RQ", "RU", "RQerr", "RUerr"), dtype=('S8', 'f8', 'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8'))

for folder in folderList:

    date = int(folder[-8:])
    
    stokes, stokesError, stokesCovar, waves, ok, bpm = fileLoad(folder)

    #extract data columns
    lam = waves[ok[0]]
    flux = stokes[0,ok[0]]
    
    if Star_type == 'johnson':
        filters = [(0, filter_B_Standard), (4, filter_V_Standard), (8, filter_R_Standard)]
    elif Star_type == 'stroemgren':
        filters = [(0, filter_B_Stroemgren), (4, filter_V_Standard), (8, filter_R_Standard) ]
    elif Star_type == 'narrow':
        filters = [(0, filter_B_WR), (4, filter_V_WR), (8, filter_R_WR)]
    else:
        print('Wrong filter type')
        break
    
    newrow = [date]
    
    for j, colour in filters:
        #extract filter info
        wave = colour['Wavelength']
        weight = colour['Filter']
        
        #Interpolate filter
        interp = interpolate.interp1d(wave, weight, bounds_error=False, fill_value=0.0)
        weightpol = interp(lam)
        filterregion = np.where(weightpol > 0)
        
        #standard columns from polsalt reduction output
        cols = [('%Q', 1), ('%U', 2)]
        errcols = [('%Qerr', 1), ('%Uerr', 2)]
        
        #cols = [('%Q', 1), ('%U', 2), ('%err', 3)]
        filter_result = []
        #repeat for each stokes value
        for column, i in cols:
            
            #load appropriate column
            pol = stokes[i, ok[0]]/stokes[0,ok[0]]
            
            #integrate and convolve
            top = np.trapz((flux[filterregion]*weightpol[filterregion]*pol[filterregion]), x=lam[filterregion])
            bottom = np.trapz((flux[filterregion]*weightpol[filterregion]), x=lam[filterregion])
            measure = top/bottom
            
            #print results
            print(column, measure)
            
            filter_result.append(measure)
        
        for column, i in errcols:
            
            #load appropriate column
            pol = stokesError[i,:]
            
            #integrate and convolve
            top = np.trapz((flux[filterregion]*weightpol[filterregion]*pol[filterregion]), x=lam[filterregion])
            bottom = np.trapz((flux[filterregion]*weightpol[filterregion]), x=lam[filterregion])
            measure = top/bottom
            
            #error calculation for error columns
            measure = measure/np.sqrt(len(lam[filterregion]))
            
            #print results
            print(column, measure)
            
            filter_result.append(measure)
        
        print(filter_result)
        newrow = np.concatenate((newrow, filter_result))
    
    print(newrow)
    final_result.add_row(newrow)    

# Write data
if Star_type == 'johnson':
    final_result.write(datapath+"_BVR_Johnson.txt", format='ascii')
if Star_type == 'stroemgren':
    final_result.write(datapath+"_Stroemgren.txt", format='ascii')
elif Star_type == 'narrow':
    final_result.write(datapath+"_bvr_narrow.txt", format='ascii')