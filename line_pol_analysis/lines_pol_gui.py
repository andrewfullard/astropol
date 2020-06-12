# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:44:30 2018

@author: Andrew
"""
import sys

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QCheckBox, QFormLayout, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QSplitter, \
QTextEdit, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Axes, Figure

import lines_pol_module as lpm

class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class MainWindow():
    def __init__(self):
        
        sys.stdout = EmittingStream(textWritten = self.normalOutputWritten)
        
        self.__lpm = lpm.LinePol()
        
        self.__loaded = False
        self.__limits = np.zeros(6)
        self.__mouseX = 0.0
        self.__lines = [None, None, None, None, None, None]
        self.__lines_Stokes = [None, None, None, None, None, None]
        self.__recorder = None
        self.__editedWidget = None

        self.__window: QMainWindow = QMainWindow()
        self.__window.setWindowTitle("Line Polarization Extractor")
        self.__widget: QWidget = QSplitter(self.__window)
        self.__window.setCentralWidget(self.__widget)
        
        self.__logOutput: QTextEdit = QTextEdit()
        
        self.__plotWidget: QWidget = QWidget()
        self.__plotWidget.setLayout(QVBoxLayout())
        
        self.__plotCanvas: FigureCanvas = FigureCanvas(Figure())
        self.__plotNavBar: NavigationToolbar = NavigationToolbar(self.__plotCanvas, self.__plotWidget)
       
        self.__plotWidget.layout().addWidget(self.__plotNavBar)
        self.__plotWidget.layout().addWidget(self.__plotCanvas)
        
        self.__rotatePA: QCheckBox = QCheckBox()
        self.__rotatePA.setText('Rotate Position Angle?')
        self.__rotatePA.stateChanged.connect(self.PAChecked)

        self.__loadButton: QPushButton = QPushButton('Load Files')   
        self.__runButton: QPushButton = QPushButton('Run Extraction') 
        self.__runAllButton: QPushButton = QPushButton('Run Extraction (all)')
        
        self.__inputPath: QLineEdit = QLineEdit()
        self.__path = self.__inputPath.text()

        self.__inputObjectName: QLineEdit = QLineEdit()
        self.__objectName = self.__inputObjectName.text()

        self.__inputPath.returnPressed.connect(self.__pathSet) 
        self.__inputObjectName.returnPressed.connect(self.__pathSet)
    
        self.__inputPA: QLineEdit = QLineEdit()
        self.__PA = self.__inputPA.text()

        self.__inputBlueMin: QLineEdit = QLineEdit()
        self.__inputBlueMin.setValidator(QIntValidator())
        self.__inputBlueMin.returnPressed.connect(self.recordLimit)
        self.__mouseBlueMin: QPushButton = QPushButton('C')
        self.__mouseBlueMin.clicked.connect(lambda: self.recordMouseLimit(self.__inputBlueMin))

        self.__inputBlueMax: QLineEdit = QLineEdit()
        self.__inputBlueMax.setValidator(QIntValidator())
        self.__inputBlueMax.returnPressed.connect(self.recordLimit)
        self.__mouseBlueMax: QPushButton = QPushButton('C')
        self.__mouseBlueMax.clicked.connect(lambda: self.recordMouseLimit(self.__inputBlueMax))

        self.__inputLineMin: QLineEdit = QLineEdit()
        self.__inputLineMin.setValidator(QIntValidator())
        self.__inputLineMin.returnPressed.connect(self.recordLimit)
        self.__mouseLineMin: QPushButton = QPushButton('C')
        self.__mouseLineMin.clicked.connect(lambda: self.recordMouseLimit(self.__inputLineMin))

        self.__inputLineMax: QLineEdit = QLineEdit()
        self.__inputLineMax.setValidator(QIntValidator())
        self.__inputLineMax.returnPressed.connect(self.recordLimit)
        self.__mouseLineMax: QPushButton = QPushButton('C')
        self.__mouseLineMax.clicked.connect(lambda: self.recordMouseLimit(self.__inputLineMax))
        
        self.__inputRedMin: QLineEdit = QLineEdit()
        self.__inputRedMin.setValidator(QIntValidator())
        self.__inputRedMin.returnPressed.connect(self.recordLimit)
        self.__mouseRedMin: QPushButton = QPushButton('C')
        self.__mouseRedMin.clicked.connect(lambda: self.recordMouseLimit(self.__inputRedMin))

        self.__inputRedMax: QLineEdit = QLineEdit()
        self.__inputRedMax.setValidator(QIntValidator())
        self.__inputRedMax.returnPressed.connect(self.recordLimit)
        self.__mouseRedMax: QPushButton = QPushButton('C')
        self.__mouseRedMax.clicked.connect(lambda: self.recordMouseLimit(self.__inputRedMax))

        self.__pathLabel: QLabel = QLabel()
        self.__pathLabel.setText("Path to objects:")

        self.__objectNameLabel: QLabel = QLabel()
        self.__objectNameLabel.setText("Object Name:")

        self.__PALabel: QLabel = QLabel()
        self.__PALabel.setText("Position Angle for rotation (degrees):")
        
        self.__blueLimitsLayout = QHBoxLayout()
        self.__lineLimitsLayout = QHBoxLayout()
        self.__redLimitsLayout = QHBoxLayout()

        self.__blueLimitsLabel: QLabel = QLabel()
        self.__blueLimitsLabel.setText("Blue limits:")

        self.__blueLimitsLayout.addWidget(self.__inputBlueMin)
        self.__blueLimitsLayout.addWidget(self.__mouseBlueMin)
        self.__blueLimitsLayout.addWidget(self.__inputBlueMax)
        self.__blueLimitsLayout.addWidget(self.__mouseBlueMax)

        self.__lineLimitsLabel: QLabel = QLabel()
        self.__lineLimitsLabel.setText("Line region:")

        self.__lineLimitsLayout.addWidget(self.__inputLineMin)
        self.__lineLimitsLayout.addWidget(self.__mouseLineMin)
        self.__lineLimitsLayout.addWidget(self.__inputLineMax)
        self.__lineLimitsLayout.addWidget(self.__mouseLineMax)

        self.__redLimitsLabel: QLabel = QLabel()
        self.__redLimitsLabel.setText("Red limits:")

        self.__redLimitsLayout.addWidget(self.__inputRedMin)
        self.__redLimitsLayout.addWidget(self.__mouseRedMin)
        self.__redLimitsLayout.addWidget(self.__inputRedMax)
        self.__redLimitsLayout.addWidget(self.__mouseRedMax)

        leftWidget: QWidget = QWidget()
        leftWidget.setLayout(QVBoxLayout())
        leftWidget.layout().addWidget(self.__pathLabel)
        leftWidget.layout().addWidget(self.__inputPath)
        leftWidget.layout().addWidget(self.__objectNameLabel)
        leftWidget.layout().addWidget(self.__inputObjectName)
        leftWidget.layout().addWidget(self.__PALabel)
        leftWidget.layout().addWidget(self.__inputPA)
        leftWidget.layout().addWidget(self.__rotatePA)

        leftWidget.layout().addWidget(self.__blueLimitsLabel)
        leftWidget.layout().addLayout(self.__blueLimitsLayout)
        leftWidget.layout().addWidget(self.__lineLimitsLabel)
        leftWidget.layout().addLayout(self.__lineLimitsLayout)
        leftWidget.layout().addWidget(self.__redLimitsLabel)
        leftWidget.layout().addLayout(self.__redLimitsLayout)

        leftWidget.layout().addWidget(self.__loadButton)
        leftWidget.layout().addWidget(self.__runButton)
        leftWidget.layout().addWidget(self.__runAllButton)

        leftWidget.layout().addWidget(self.__logOutput)
        
        self.__widget.addWidget(leftWidget)
        self.__widget.addWidget(self.__plotWidget)
        
        self.__axes: Axes = self.__plotCanvas.figure.add_subplot(211)
        self.__axes.set_ylabel("$F_\lambda$")
        self.__axes_Stokes: Axes = self.__plotCanvas.figure.add_subplot(212)
        self.__axes_Stokes.set_ylabel("$Q$")
        self.__axes_Stokes.set_xlabel("$\lambda (\AA)$")

        print("Enter path to observations, object name, then press enter. Select position angle rotation if desired.\nPress Load Files, then choose your limits on the plot. Press Run Extraction.")
        
    def __del__(self):
        '''Restore sys.stdout'''
        sys.stdout = sys.__stdout__
     
    def getWindow(self) -> QMainWindow:
        '''Returns window'''
        return self.__window
    
    def __pathSet(self) -> None:
        '''Detects button presses and sets variables from text boxes'''
        self.__path = self.__inputPath.text()
        self.__objectName = self.__inputObjectName.text()
        if (self.__path is not None) and (self.__objectName is not None):
            self.__loadButton.clicked.connect(self.loadOneFile)
            self.__runButton.clicked.connect(self.__lpm.doLinePolExtractionSequence)
            self.__runAllButton.clicked.connect(self.__lpm.doLinePolExtractionAll)
            print('Input path: ', self.__path)
            print('Input object: ', self.__objectName)

    def loadOneFile(self):
        '''Loads and plots the first spectrum'''
        self.__lpm.loadOneFile(self.__path, self.__objectName)
        #clear axes
        self.__axes.cla()
        self.__axes_Stokes.cla()
        #replot
        self.__axes.set_ylabel("$F_\lambda$")
        self.__axes_Stokes.set_ylabel("$Q$")
        self.__axes_Stokes.set_xlabel("$\lambda (\AA)$")

        self.__axes.plot(self.__lpm.wavelengths, self.__lpm.i)
        self.__axes_Stokes.plot(self.__lpm.wavelengths, self.__lpm.q)
        self.plotvLines(self.__limits)
        self.__plotCanvas.draw_idle()
        self.__loaded = True
        
    def normalOutputWritten(self, text) -> None:
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.__logOutput.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.__logOutput.setTextCursor(cursor)
        self.__logOutput.ensureCursorVisible()
        
    def PAChecked(self) -> None:
        '''Records and outputs PA options'''
        if self.__rotatePA.isChecked() == True:
            self.__lpm.rotation = True
            self.__PA = self.__inputPA.text()
            print('PA will be rotated by ', self.__PA, ' deg')
        else:
            self.__lpm.rotation = False
            print('PA will not be rotated')

    def getMousePos(self, event) -> None:
        '''Gets and plots mouse position'''
        if event.button == 1:
            self.__mouseX = int(event.xdata)
            self.__editedWidget.setText(str(self.__mouseX))
            self.__plotCanvas.mpl_disconnect(self.__recorder)
            self.recordLimit()
        else:
            pass
        
    def plotvLines(self, xList) -> None:
        '''Plots and updates vertical lines to show limits'''
        if self.__loaded:
            for i, x in enumerate(xList):
                if self.__lines[i] is not None:
                    self.__lines[i].remove()
                
                if self.__lines_Stokes[i] is not None:
                    self.__lines_Stokes[i].remove()

                if x > 0:
                    self.__lines[i] = self.__axes.vlines(x, 0, np.max(self.__lpm.i))
                    self.__lines_Stokes[i] = self.__axes_Stokes.vlines(x, np.min(self.__lpm.q), np.max(self.__lpm.q))

                self.__plotCanvas.draw_idle()
    
    def recordLimit(self) -> None:
        '''Records limits to variables'''
        if len(self.__inputBlueMin.text()) > 0:
            print("Blue continuum min set to: ", self.__inputBlueMin.text())
            self.__limits[0] = self.__lpm.blueContinuumMin = float(self.__inputBlueMin.text())
        
        if len(self.__inputBlueMax.text()) > 0:
            print("Blue continuum max set to: ", self.__inputBlueMax.text())
            self.__limits[1] = self.__lpm.blueContinuumMax = float(self.__inputBlueMax.text())
        
        if len(self.__inputLineMin.text()) > 0:
            print("Line min set to: ", self.__inputLineMin.text())
            self.__limits[2] = self.__lpm.lineMin = float(self.__inputLineMin.text())
        
        if len(self.__inputLineMax.text()) > 0:
            print("Line max set to: ", self.__inputLineMax.text())
            self.__limits[3] = self.__lpm.lineMax = float(self.__inputLineMax.text())
        
        if len(self.__inputRedMin.text()) > 0:
            print("Red continuum min set to: ", self.__inputRedMin.text())
            self.__limits[4] = self.__lpm.redContinuumMin = float(self.__inputRedMin.text())

        if len(self.__inputRedMax.text()) > 0:
            print("Red continuum max set to: ", self.__inputRedMax.text())
            self.__limits[5] = self.__lpm.redContinuumMax = float(self.__inputRedMax.text())

        #Once all inputs are given, check for errors
        if len(self.__inputBlueMin.text()) > 0 and len(self.__inputBlueMax.text()) > 0 and len(self.__inputLineMin.text()) > 0 \
            and len(self.__inputLineMax.text()) > 0 and len(self.__inputRedMin.text()) > 0 and len(self.__inputRedMax.text()) > 0:
            self.__lpm.wavelengthErrorCheck()

        #Plot the limits
        self.plotvLines(self.__limits)
    
    def recordMouseLimit(self, widget) -> None:
        '''Starts mouse connection'''
        self.__recorder = self.__plotCanvas.mpl_connect('button_press_event', self.getMousePos)
        self.__editedWidget = widget
        print("Click on the plot to set a limit")