from PyQt5 import QtGui, QtWidgets, QtCore
import numpy as np
import sys
import math
from gui import Ui_PlotForm
import math
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

class ApplicationWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_PlotForm()
        self.ui.setupUi(self)
        self.time  = np.arange(0,1,0.001) #in sec but step in 1 msec
        self.vector= np.matrix ([0,0,1]) #da range sabt

        pixelIntensity = 255 #depend on pixel choosen

        self.T1 = self.createT1(pixelIntensity)
        self.T2 = self.createT2(pixelIntensity)
        self.PD = self.createPD(pixelIntensity)
        

        self.ui.rotationAngle.textChanged.connect(self.plot)  #any change in lineEdit text
       
        self.ui.te.textChanged.connect(self.plot)  #any change in lineEdit text
        self.ui.tr.textChanged.connect(self.plot)  #any change in lineEdit text
       
        self.ui.Plot.clicked.connect(self.plot)
    
    def plot(self):
        self.ui.decayMx.clear()
        self.ui.decayMy.clear()
        self.ui.recoveryMz.clear()
        
        self.DecayMx = self.ui.decayMx
        self.DecayMy = self.ui.decayMy
        self.RecoveryMz = self.ui.recoveryMz


        self.theta = ((float) (self.ui.rotationAngle.text())) #5ly balk not global 
        self.Tr = ((float) (self.ui.tr.text()))
        self.Te = ((float) (self.ui.te.text()))
        

        speed =1 

        self.Mx = []
        self.My = []
        self.Mz =[]
        
        self.vector = self.rotationAroundYaxisMatrix(self.theta,self.vector)

        for i in range(len(self.time)):
            self.vector = self.rotationAroundZaxisMatrixXY(self.Tr,speed,self.vector,self.time[i])
            self.vector = self.recoveryDecayEquation(self.T1,self.T2,self.PD, self.vector,self.time[i])
            
            self.Mx = np.append(self.Mx,self.vector.item(0))
            self.My = np.append(self.My,self.vector.item(1))
            self.Mz = np.append(self.Mz,self.vector.item(2))
        
    

        self.DecayMx.plot(self.time,np.ravel(self.Mx))
        self.DecayMy.plot(self.time,np.ravel(self.My))
        self.RecoveryMz.plot(self.time,np.ravel(self.Mz))

        self.RecoveryMz.addLine(x=self.Tr)
        self.RecoveryMz.addLine(x=self.Te)
        self.DecayMx.addLine(x=self.Tr)
        self.DecayMx.addLine(x=self.Te)

        
    def createPD(self,intensity):
        return (1/255)*intensity 
        
    def createT1 (self,intensity):
        return ((6*intensity)+500)/1000

    def createT2(self,intensity):
        return ((2*intensity)+20)/1000
    def returnIntensity(self,Pd): # proton intensity vales from 0 till 1 
        return 255*Pd
    

    
    def mappingT1 (self,T1): #T1 in msec assumption
        return (T1-500)/6

    def mappingT2 (self,T2):  #T1 in msec assumption
        return (T2-20)/2

    def rotationAroundYaxisMatrix(self,theta,vector):
            vector = vector.transpose()
            theta = (math.pi / 180) * theta
            R = np.matrix ([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]] )
            R = np.dot(R, vector)
            R = R.transpose()
            return np.matrix(R)


    def rotationAroundZaxisMatrixXY(self,TR,speed,vector,time): #time = self.time
            vector = vector.transpose()
            theta = speed * (time/ TR)
            theta = (math.pi / 180) * theta
            XY = np.matrix([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0, 0, 1]])
            XY = np.dot(XY,vector)
            XY = XY.transpose()
            return np.matrix(XY) 


    def recoveryDecayEquation(self,T1,T2,PD,vector,time):
            vector = vector.transpose()
            Decay =np.matrix([[np.exp(-time/T2),0,0],[0,np.exp(-time/T2),0],[0,0,np.exp(-time/T1)]])
            Decay = np.dot(Decay,vector)
        
            Rec= np.dot(np.matrix([[0,0,(1-(np.exp(-time/T1)))]]),PD)
            Rec = Rec.transpose()
            Decay = np.matrix(Decay)
            Rec =  np.matrix(Rec)    
        
            RD  = Decay + Rec
            RD = RD.transpose()
            return RD





app = QtWidgets.QApplication(sys.argv)
application = ApplicationWindow()
application.show()
app.exec_()


