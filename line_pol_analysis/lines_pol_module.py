# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:07:47 2018

@author: Andrew
"""

#----------------------------------------------------------------
# Program to measure polarization, q, u and theta, with errors, for an
# emission line.   JRL, 4/13/2010
# Will rotate q and u values to a user specified angle and accepts
# two input files for 1 spectra (ie. HPOL blue and red CCD files). 
# Also still accepts only one file.     JRL, 6/10/2010
# Accepts two files, does errors correctly.    JRL, 6/23/2010
# Fixed problem with not calculating underlying absorption
# correctly. (EW inputed by user needs to be negative, but I
# ran it through the math as a positive number before). JRL, 3/26/2013
# 
# 170722 jlh modified for RSS data
# 82018 agf written in Python
#-----------------------------------------------------------------

import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table
import glob
import numpy as np
from scipy import interpolate

class LinePol():
    def __init__(self):
        #-----------------------------------------------------------------
        # Pixel correlation value (different for each detector)
        #----------------------------------------------------------------_
        self.__pixelCorrelation = 1.0  #RSS
        self.__folderPath = str()
        self.__folderList = str()            
        self.__path = str()
        self.__objectName = str()
        self.__currentFolderIndex = 0
        self.__outputTable = None

        self.__filePattern = '*.fits'
        self.__positionAngleRotate = 0.0
        self.__absorptionDeltaWave = 0.0
        
        self.__indexBlueContinuumMin = 0
        self.__indexBlueContinuumMax = 0
        self.__indexRedContinuumMin = 0
        self.__indexRedContinuumMax = 0
        self.__indexLineMin = 0
        self.__indexLineMax = 0
        
        self.__closestBlueContinuumMin = 0.0
        self.__closestBlueContinuumMax = 0.0
        self.__closestRedContinuumMin = 0.0
        self.__closestRedContinuumMax = 0.0
        self.__closestLineMin = 0.0
        self.__closestLineMax = 0.0
        
        self.blueContinuumMin = 0.0
        self.blueContinuumMax = 0.0
        self.redContinuumMin = 0.0
        self.redContinuumMax = 0.0
        self.lineMin = 0.0
        self.lineMax = 0.0
        
        self.meanBlueContinuumLam = 0.0
        self.meanRedContinuumLam = 0.0
        
        self.meanQBlueContinuum = 0.0
        self.meanUBlueContinuum = 0.0
        
        self.meanQRedContinuum = 0.0
        self.meanURedContinuum = 0.0
        
        self.meanErrBlueContinuum = 0.0
        self.meanErrRedContinuum = 0.0
        
        self.meanLineLam = 0.0
        self.meanQLine = 0.0
        self.meanULine = 0.0
        self.meanErrLine = 0.0

        self.blueContinuumCenter = 0.0
        self.redContinuumCenter = 0.0        
        
        self.wavelengths = np.ndarray(0)
        
        self.i = np.ndarray(0)
        self.q = np.ndarray(0)
        self.u = np.ndarray(0)
        
        self.iErr = np.ndarray(0)
        self.qErr = np.ndarray(0)
        self.uErr = np.ndarray(0)
        
        self.iMeanTotal = 0.0
        self.qMeanTotal = 0.0
        self.uMeanTotal = 0.0
        self.errMeanTotal = 0.0     
        
        self.iLineFlux = 0.0
        self.qLineFlux = 0.0
        self.uLineFlux = 0.0
        self.errLineFlux = 0.0
        
        self.qContOutput = 0.0
        self.uContOutput = 0.0
        self.errContOutput = 0.0
        self.pContOutput = 0.0
        self.PAContOutput = 0.0
        
        self.qLineOutput = 0.0
        self.uLineOutput = 0.0
        self.errLineOutput = 0.0
        self.pLineOutput = 0.0
        self.PALineOutput = 0.0
        
        self.qDiffOutput = 0.0
        self.uDiffOutput = 0.0
        self.errDiffOutput = 0.0
        
        self.rotation = False
        
    def loadOneFile(self, path, objectName):
        '''Loads a file for display'''
        self.getInput(path, objectName)
        if (len(self.__folderList) > 0) and (self.__currentFolderIndex < len(self.__folderList)):
            self.fileLoad(self.__folderList[self.__currentFolderIndex])
        else:
            print("No files found at location: ", path)

    def doLinePolExtraction(self, folder): 
        '''Runs line polarization extraction on one folder'''
        self.fileLoad(folder)    
        
        if self.rotation == True:
            self.PARotation()
        
        self.setClosestValues()
        
        waveBlueContinuum = self.sliceWavelengths(self.wavelengths, self.__indexBlueContinuumMin, self.__indexBlueContinuumMax)
        iBlueContinuum = self.sliceWavelengths(self.i, self.__indexBlueContinuumMin, self.__indexBlueContinuumMax)
        qBlueContinuum = self.sliceWavelengths(self.q, self.__indexBlueContinuumMin, self.__indexBlueContinuumMax)
        uBlueContinuum = self.sliceWavelengths(self.u, self.__indexBlueContinuumMin, self.__indexBlueContinuumMax)
        errBlueContinuum = self.sliceWavelengths(self.qErr, self.__indexBlueContinuumMin, self.__indexBlueContinuumMax)
        
        waveRedContinuum = self.sliceWavelengths(self.wavelengths, self.__indexRedContinuumMin, self.__indexRedContinuumMax)
        iRedContinuum = self.sliceWavelengths(self.i, self.__indexRedContinuumMin, self.__indexRedContinuumMax)
        qRedContinuum = self.sliceWavelengths(self.q, self.__indexRedContinuumMin, self.__indexRedContinuumMax)
        uRedContinuum = self.sliceWavelengths(self.u, self.__indexRedContinuumMin, self.__indexRedContinuumMax)
        errRedContinuum = self.sliceWavelengths(self.qErr, self.__indexRedContinuumMin, self.__indexRedContinuumMax)
        
        waveLine = self.sliceWavelengths(self.wavelengths, self.__indexLineMin, self.__indexLineMax)
        iLine = self.sliceWavelengths(self.i, self.__indexLineMin, self.__indexLineMax)
        qLine = self.sliceWavelengths(self.q, self.__indexLineMin, self.__indexLineMax)
        uLine = self.sliceWavelengths(self.u, self.__indexLineMin, self.__indexLineMax)
        errLine = self.sliceWavelengths(self.qErr, self.__indexLineMin, self.__indexLineMax)
        
        lineCenter = self.findCenter(waveLine)
        blueContinuumCenter = self.findCenter(waveBlueContinuum)
        redContinuumCenter = self.findCenter(waveRedContinuum)
        
        self.meanBlueContinuumLam = np.mean(iBlueContinuum)
        self.meanRedContinuumLam = np.mean(iRedContinuum)
        
        self.meanQBlueContinuum = np.mean(qBlueContinuum)
        self.meanUBlueContinuum = np.mean(uBlueContinuum)
        
        self.meanQRedContinuum = np.mean(qRedContinuum)
        self.meanURedContinuum = np.mean(uRedContinuum)
        
        self.meanErrBlueContinuum = self.findErrorAverage(errBlueContinuum)
        self.meanErrRedContinuum = self.findErrorAverage(errRedContinuum)
        
        self.meanLineLam = np.mean(iLine)
        self.meanQLine = np.mean(qLine)
        self.meanULine = np.mean(uLine)
        self.meanErrLine = self.findErrorAverage(errLine)
        
        errorWeight = self.findErrorWeight(waveLine, lineCenter, waveBlueContinuum, blueContinuumCenter, waveRedContinuum, redContinuumCenter)
        
        self.iMeanTotal = self.findTotalAverage(waveBlueContinuum, self.meanBlueContinuumLam, blueContinuumCenter, \
                                                waveRedContinuum, self.meanRedContinuumLam, redContinuumCenter, \
                                                waveLine, lineCenter)
        self.qMeanTotal = self.findTotalAverage(waveBlueContinuum, self.meanQBlueContinuum, blueContinuumCenter, \
                                                waveRedContinuum, self.meanQRedContinuum, redContinuumCenter, \
                                                waveLine, lineCenter)
        self.uMeanTotal = self.findTotalAverage(waveBlueContinuum, self.meanUBlueContinuum, blueContinuumCenter, \
                                                waveRedContinuum, self.meanURedContinuum, redContinuumCenter, \
                                                waveLine, lineCenter)
        self.errMeanTotal = self.findErrorWeightedAverage(errorWeight, self.meanErrBlueContinuum, self.meanErrRedContinuum)
        
        lineWidth = self.findLineWidth(self.lineMin, self.lineMax)
        
        self.iLineFlux = self.calcLineFlux(self.meanLineLam, self.iMeanTotal, lineWidth)
        self.qLineFlux = self.calcLineFlux(self.meanQLine, self.qMeanTotal, lineWidth)
        self.uLineFlux = self.calcLineFlux(self.meanULine, self.uMeanTotal, lineWidth)
        self.errLineFlux = self.calcLineFluxError(self.meanErrLine, self.errMeanTotal, lineWidth)
        
        self.qContOutput = self.qMeanTotal / self.iMeanTotal * 100
        self.uContOutput = self.uMeanTotal / self.iMeanTotal * 100
        self.errContOutput = self.errMeanTotal / self.iMeanTotal * 100
        self.pContOutput = self.calcPolarization(self.qContOutput, self.uContOutput)
        self.PAContOutput = self.calcPA(self.qContOutput, self.uContOutput)
        
        self.qLineOutput = self.qLineFlux / self.iLineFlux * 100
        self.uLineOutput = self.uLineFlux / self.iLineFlux * 100
        self.errLineOutput = self.errLineFlux / self.iLineFlux * 100
        self.pLineOutput = self.calcPolarization(self.qLineOutput, self.uLineOutput)
        self.PALineOutput = self.calcPA(self.qLineOutput, self.uLineOutput)
        
        self.qDiffOutput = (self.qContOutput - self.qLineOutput) * 100
        self.uDiffOutput = (self.uContOutput - self.uLineOutput) * 100
        #not actually calculating polarization here, but does the same thing
        self.errDiffOutput = self.calcPolarization(self.errContOutput, self.errLineOutput)
        
        self.printOutput(folder)
        self.addToTable()
        
        return lineCenter
        
    def doLinePolExtractionAll(self): 
        '''Runs line polarization extraction automatically for all data'''
        self.constructOutputTable()
        
        lineCenter = 0
        
        for folder in self.__folderList:
            lineCenter = self.doLinePolExtraction(folder)
            
        self.writeTable(lineCenter)    
        
    def doLinePolExtractionSequence(self): 
        '''Runs line polarization extraction for one observation and then move to the next'''
        if not self.__outputTable:
            self.constructOutputTable()
        
        lineCenter = 0

        if self.__currentFolderIndex > (len(self.__folderList) - 1):
            print("End of file list")
            self.__outputTable = None
            return

        lineCenter = self.doLinePolExtraction(self.__folderList[self.__currentFolderIndex])        
        self.__currentFolderIndex += 1
            
        if self.__currentFolderIndex == (len(self.__folderList)):
            self.writeTable(lineCenter)

        return
        
    def valueLocate(self, array, value):
        '''Locates nearest value in array'''
        index = (np.abs(array - value)).argmin()
        output = array[index]
        return output, index
    
    def IDLInterpol(self, inputArray, inputAbscissa, outputAbscissa):
        '''Wrapper for scipy interpolate to match IDL style'''
        interpfunc = interpolate.interp1d(inputAbscissa, inputArray, kind='linear')
        return interpfunc(outputAbscissa)
       
    def constructOutputTable(self) -> None:
        '''Sets up the output astropy table'''
        self.__outputTable = Table(names = ["Date", "%Q", "%U", "%Err", "%P", "PA"])
    
        self.__outputTable["%Q"].format = "{:.5f}"
        self.__outputTable["%U"].format = "{:.5f}"
        self.__outputTable["%Err"].format = "{:.7f}"
        self.__outputTable["%P"].format = "{:.3f}"
        self.__outputTable["PA"].format = "{:.1f}"

    def getInput(self, path, objectName) -> None:
        '''Gets file and folder locations'''
        #get path to folder 
        self.__path = path
        #pick star
        self.__objectName = objectName
        #find dated folders using dropbox naming format
        self.__folderPath = self.__path+"/"+self.__objectName+'/20*/'
        self.__folderList = glob.glob(self.__folderPath)
        #fits file search pattern

    def wavelengthErrorCheck(self) -> None:
        '''If continuum extends into the line ask for values again.'''
        if (self.blueContinuumMin > self.blueContinuumMax) or (self.lineMin > self.lineMax) or (self.redContinuumMin > self.redContinuumMax):
            print('Error! Min < Max')
        
        if self.lineMin < self.blueContinuumMax:
            print('Error! Continuum extends into line region. Please reenter values.')

        if self.redContinuumMin < self.lineMax:
            print('Error! Continuum extends into line region. Please reenter values.')

    def fileLoad(self, folder) -> None:
        '''Loads a fits file'''
        dataFile = glob.glob(folder+self.__filePattern)

        #Open fits file
        hdul = fits.open(dataFile[0])
        #get wavelength spacing
        deltaWave = float(hdul['SCI'].header['CDELT1'])
        #get starting wavelength
        wave0 = float(hdul['SCI'].header['CRVAL1'])
        #get wavelength axis size
        waves = int(hdul['SCI'].header['NAXIS1'])

       # stokes I, Q, U values
        stokesSw = hdul['SCI'].data[:,0,:]
        #stokes errors
        varSw = hdul['VAR'].data[:,0,:]
        #wavelength axis
        self.wavelengths = wave0 + deltaWave*np.arange(waves)
    
        print("\n"+folder)
        
        self.i = stokesSw[0, :]
        self.q = stokesSw[1, :]#[i > 0]/i[i > 0]
        self.u = stokesSw[2, :]#[i > 0]/i[i > 0]
        
        self.iErr = np.sqrt(varSw[0, :])
        self.qErr = np.sqrt(varSw[1, :])#[iErr > 0])/i[i > 0]
        self.uErr = np.sqrt(varSw[2, :])#[iErr > 0])/i[i > 0]

    def PARotation(self) -> None:
        '''Rotate the data in a file if need be. Errors do not need to be rotated since they will be essentially the same.'''
        positionAngleArray = np.rad2deg(0.5*np.arctan2(self.q, self.u))
        deltaPositionAngle = []
        qRotated = []
        uRotated = []
        
        for i in range(len(positionAngleArray)):
            if positionAngleArray[i] < 0:
                positionAngleArray[i] += 180
        
        polarization = np.sqrt(self.q**2 + self.u**2)
        
        #Find the angle to rotate by and then convert that angle to radians so you can use sine and cosine later.
        for angle in positionAngleArray:
            deltaPositionAngle.append(np.deg2rad(angle - self.__positionAngleRotate))
        
        #compute q values for the rotated data  
        for i in range(len(deltaPositionAngle)):
            qRotated.append(polarization[i] * np.cos(2 * deltaPositionAngle[i]))
            
        #compute u values for the rotated data                     
        for i in range(len(deltaPositionAngle)):
            uRotated.append(polarization[i] * np.cos(2 * deltaPositionAngle[i]))
            
        self.q = qRotated
        self.u = uRotated
            
    def setClosestValues(self) -> None:
        '''Find the wavelength values closest to C1, C2, L1, L2, C3 and C4.'''  
        self.__closestBlueContinuumMin, self.__indexBlueContinuumMin = self.valueLocate(self.wavelengths, self.blueContinuumMin)
        self.__closestBlueContinuumMax, self.__indexBlueContinuumMax = self.valueLocate(self.wavelengths, self.blueContinuumMax)
        self.__closestLineMin, self.__indexLineMin = self.valueLocate(self.wavelengths, self.lineMin)
        self.__closestLineMax, self.__indexLineMax = self.valueLocate(self.wavelengths, self.lineMax)
        self.__closestRedContinuumMin, self.__indexRedContinuumMin = self.valueLocate(self.wavelengths, self.redContinuumMin)
        self.__closestRedContinuumMax, self.__indexRedContinuumMax = self.valueLocate(self.wavelengths, self.redContinuumMax)

    def sliceWavelengths(self, wavelengths, indexMin, indexMax) -> None:
        '''Pull out the wavelengths'''
        return wavelengths[indexMin:indexMax]
        
    def findCenter(self, wave) -> float:
        '''Find center.'''
        center = len(wave)/2      
        #later I subtract one from the center values. This is because if the center
        #is 3 that means it is element 2 in the array (0,1,2,...). However if it is element
        #one element array than 1/2=0 (integers) so then when I subtract later on I get -1.
        #Here I correct for that.  
        
        if center == 0: center = 1

        return center
        
    def findErrorAverage(self, err) -> float:
        '''Finds the average error'''
        return np.sqrt(np.sum((err)**2.) * self.__pixelCorrelation / (len(err)**2))
        
    def findErrorWeight(self, waveLine, lineCenter, waveBlueContinuum, blueContinuumCenter, waveRedContinuum, redContinuumCenter) -> float:
        '''Find how much to weight the blue continuum region by. The red is
        one minus this value. This is needed instead of the interpolate 
        function, which does not work for the errors.'''
            
        return (waveLine[int(lineCenter) - 1] - waveBlueContinuum[int(blueContinuumCenter) - 1]) \
                        /(waveRedContinuum[int(redContinuumCenter) - 1] - waveBlueContinuum[int(blueContinuumCenter) - 1])
                        
    def findTotalAverage(self, waveBlueContinuum, meanBlueContinuum, blueContinuumCenter, waveRedContinuum, meanRedContinuum, redContinuumCenter, waveLine, lineCenter) -> None:
        '''Find total average of both regions'''
        return self.IDLInterpol([meanBlueContinuum, meanRedContinuum], [waveBlueContinuum[int(blueContinuumCenter) - 1], \
                                 waveRedContinuum[int(redContinuumCenter) - 1]], waveLine[int(lineCenter) - 1])
        
    def findErrorWeightedAverage(self, errorWeight, meanErrBlueContinuum, meanErrRedContinuum) -> float:
        '''Finds the error weighted average'''
        return np.sqrt((errorWeight * meanErrBlueContinuum)**2 + ((1 - errorWeight) * meanErrRedContinuum)**2)
   
    def findLineWidth(self, lineMin, lineMax) -> float:
        '''Calculate line width here. It 
        goes into the equation used to calculate the line pol.'''
        return lineMax - lineMin
    
    def calcLineFlux(self, mean, total, lineWidth) -> float:
        '''Calculate the flux in the line.'''
        return (mean - total) * lineWidth
    
    def calcLineFluxError(self, mean, total, lineWidth) -> float:
        '''Calculate the flux error in the line.'''
        return np.sqrt(mean**2 + total**2) * lineWidth
    
    def contOutput(self, stokesMeanTotal, iMeanTotal) -> float:
        '''Continuum percentage output'''
        return stokesMeanTotal / iMeanTotal * 100
    
    def calcPolarization(self, q, u) -> float:
        '''Calculates total polarization'''
        return np.sqrt(q**2 + u**2)

    def calcPA(self, q, u) -> float:
        '''Calculates position angle'''
        return np.rad2deg(0.5 * np.arctan2(u, q))
    
    def addToTable(self) -> None:
        '''Adds a new row to the output table'''
        newrow = [self.date, self.qLineOutput, self.uLineOutput, self.errLineOutput, self.pLineOutput, self.PALineOutput]
        self.__outputTable.add_row(newrow)
    
    def writeTable(self, lineCenter) -> None:
        '''Writes the output table to file'''
        self.__outputTable.write(self.__objectName+'_'+str(lineCenter+self.lineMin)+'.txt', format='ascii', overwrite=True)           
    
    def printOutput(self, folder) -> None:
        '''Prints output to console'''
        self.date = folder[-10:]
        self.date = self.date.replace("\\", "")
        
        print('Date: ', self.date, '\n')
        print('C1 ', 'C2 ', 'L1 ', 'L2 ', 'C3 ', 'C4')
        print(self.__closestBlueContinuumMin, self.__closestBlueContinuumMax, self.__closestLineMin, self.__closestLineMax, self.__closestRedContinuumMin, self.__closestRedContinuumMax)
        print( ' ', ' - ', 'I ', 'Q ', 'U ', 'Err ', '%Pol ', 'PA ')
        print('Flam Left ', self.meanBlueContinuumLam,' ', self.meanQBlueContinuum,' ', self.meanUBlueContinuum ,' ', self.meanErrBlueContinuum,' ', \
              self.calcPolarization(self.meanQBlueContinuum, self.meanUBlueContinuum),' ', self.calcPA(self.meanQBlueContinuum, self.meanUBlueContinuum))
        print('Flam Cntr ', self.meanLineLam,' ', self.meanQLine,' ', self.meanULine,' ', self.meanErrLine,' ', self.calcPolarization(self.meanQLine, self.meanULine),' ', \
              self.calcPA(self.meanQLine, self.meanULine))
        print('Flam Right ', self.meanRedContinuumLam,' ', self.meanQRedContinuum,' ', self.meanURedContinuum ,' ', self.meanErrRedContinuum,' ', \
              self.calcPolarization(self.meanQRedContinuum, self.meanURedContinuum),' ', self.calcPA(self.meanQRedContinuum, self.meanURedContinuum))
        print('Flam Cont ', self.iMeanTotal,' ',self.qMeanTotal,' ', self.uMeanTotal ,' ', self.errMeanTotal,' ', \
              self.calcPolarization(self.qMeanTotal, self.uMeanTotal),' ', self.calcPA(self.qMeanTotal, self.uMeanTotal))
        print('Flux Line ', self.iLineFlux,' ', self.qLineFlux,' ', self.uLineFlux,' ', self.errLineFlux,' ', \
              self.calcPolarization(self.qLineFlux, self.uLineFlux),' ', self.calcPA(self.qLineFlux, self.uLineFlux))
        print('EW Line ', self.iLineFlux/self.iMeanTotal,' ', self.qLineFlux*(self.iLineFlux/self.iMeanTotal)/self.iLineFlux,' ', self.uLineFlux*(self.iLineFlux/self.iMeanTotal)/self.iLineFlux,' ',\
              self.errLineFlux*(self.iLineFlux/self.iMeanTotal)/self.iLineFlux,' ', np.sqrt((self.qLineFlux*(self.iLineFlux/self.iMeanTotal)/self.iLineFlux)**2 + (self.uLineFlux*(self.iLineFlux/self.iMeanTotal)/self.iLineFlux)**2),' ')
        print('% Cont ', ' - ', self.qContOutput,' ', self.uContOutput,' ', self.errContOutput,' ', self.pContOutput,' ', self.PAContOutput)
        print('% Line ', ' - ', self.qLineOutput,' ', self.uLineOutput,' ', self.errLineOutput,' ', self.pLineOutput,' ', self.PALineOutput)
        print('% Cnt-Line ', ' - ', self.qDiffOutput,' ', self.uDiffOutput,' ', self.errDiffOutput)