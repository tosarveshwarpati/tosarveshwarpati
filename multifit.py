import pylab as plb
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

with open('alpha.csv', 'r') as i:
    rawdata = list(csv.reader(i), delimiter=',')

#data = np.array(rawdata[1:], dtype = np.float )
#x = data[:,0]
#y = data[:,1]
#data = np.loadtxt("alpha.csv", delimiter=",")
#x = data[0,:]
#y = data[1,:]
#
offset = input('input your offset(input 0 for null offset)')


def gauss3(x,a1,mean1,sigma1,a2,mean2,sigma2,a3,mean3,sigma3,offset):
    return {a1*(1/(sigma1*(numpy.sqrt(2*numpy.pi))))*numpy.exp(-(x-mean1)**2/(2*sigma1**2))}+{a2*(1/(sigma2*(numpy.sqrt(2*numpy.pi))))*numpy.exp(-(x-mean2)**2/(2*sigma2**2))}+{a3*(1/(sigma3*(numpy.sqrt(2*numpy.pi))))*numpy.exp(-(x-mean3)**2/(2*sigma3**2))} + offset

    
def gauss2(x,a1,mean1,sigma1,a2,mean2,sigma2,offset):
    return gauss3(x,a1,mean1,sigma1,a2,mean2,sigma2,0,0,0,offset = 0) + offset  
       

def gauss1(x,a1,mean1,sigma1,offset):
    return gauss3(x,a1,mean1,sigma1,0,0,0,0,0,0,offset = 0) + offset  

    
def lorentz3(x, a1, cen1, width1, a2, cen2, width2, a3, cen3, width3, offset):
    return {a1*width1**2/(x-cen1)**2+width1**2} + {a2*width2**2/(x-cen2)**2+width2**2} + {a3*width3**2/(x-cen3)**2+width3**2} + offset

    
def lorentz2(x, a1, cen1, width1, a2, cen2, width2, offset):
    return lorentz3(x, a1, cen1, width1, a2, cen2, width2, 0, 0, 0, offset = 0) + offset    


def lorentz1(x, a1, cen1, width1, offset):
    return lorentz3(x, a1, cen1, width1, 0, 0, 0, 0, 0, 0, offset = 0) + offset    


def voigt3(x,ag1,mg1,sg1,al1,cl1,wl1,ag2,mg2,sg2,al2,cl2,wl2,ag3,mg3,sg3,al3,cl3,wl3,offset):
    return {(ag1*(1/(sg1*numpy.sqrt(2*numpy.pi))))*(numpy.exp(-(x-mg1)**2))/((2*sg1)**2)} + {(al1*wl1**2)/((x-cl1)**2+wl1**2)} + {(ag2*(1/(sg2*numpy.sqrt(2*numpy.pi))))*(numpy.exp(-(x-mg2)**2))/((2*sg2)**2)} + {(al2*wl2**2)/((x-cl2)**2+wl2**2)} + {(ag3*(1/(sg3*numpy.sqrt(2*numpy.pi))))*(numpy.exp(-(x-mg3)**2))/((2*sg3)**2)} + {(al3*wl3**2)/((x-cl3)**2+wl3**2)} + offset


def voigt2(x,ag1,mg1,sg1,al1,cl1,wl1,ag2,mg2,sg2,al2,cl2,wl2,offset):
    return voigt3(x,ag1,mg1,sg1,al1,cl1,wl1,ag2,mg2,sg2,al2,cl2,wl2,0,0,0,0,0,0,offset = 0) + offset


def voigt1(x,ag1,mg1,sg1,al1,cl1,wl1,offset):
    return voigt3(x,ag1,mg1,sg1,al1,cl1,wl1,0,0,0,0,0,0,0,0,0,0,0,0,offset = 0) + offset
    
print("%10.3e"% (356.08977))
print("Enter the lineshape profile")
print("a for Gaussian")
print("b for Lorentzian")
print("c for Voigt")

while True:
 profile = input("Enter choice(a/b/c): ")
if profile in ('a', 'b', 'c'):
     print("LINE PROFILE is selected")
 
     if profile == 'a':
         print("Enter the peak count 1/2/3")
         count = input("number of peaks for fitting")
         if count in ('1', '2', '3'):
         
        
                   if  count == '1':
                       popt, pcov = scipy.optimize.curve_fit(gauss1, x, ydata, p0 = [a1, mean1, sigma1])
                       perr = numpy.sqrt(numpy.diag(pcov))
                       
                       print ("hight parameter = %0.4f (+/-) %0.4f" % (popt[0], perr[0]))
                       print ("mean = %0.4f (+/-) %0.4f" % (popt[1], perr[1]))
                       print ("variance = %0.4f (+/-) %0.4f" % (popt[2], perr[2]))
                       
                       
                   elif count == '2':
              
                   elif count == '3':
              
                       break
         
         else: 
                   print("invalid selection of peak count")
                  
                  
                  
         
     elif profile == 'b':
            print("Enter the peak count 1/2/3")
            count = input("number of peaks for fitting")
            if count in ('1', '2', '3'):
         
        
                   if  count == '1':
                       
                   elif count == '2':
              
                   elif count == '3':
              
                    break
         
            else: 
                  print("invalid selection of peak count")
                  
             
         

     elif profile == 'c':
            print("Enter the peak count 1/2/3")
            count = input("number of peaks for fitting")
            if count in ('1', '2', '3'):
         
        
                   if  count == '1':
                       
                   elif count == '2':
              
                   elif count == '3':
              
                   break
         
            else: 
                  print("invalid selection of peak count")
                  
         
         
         
         
     break
else:
     print("Invalid Input")


