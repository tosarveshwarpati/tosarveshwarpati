import numpy as np
import scipy.signal
import scipy
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.pyplot import figure, show
from matplotlib.widgets import Slider, Button
import matplotlib.backend_bases
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True
import sys
T=1
lambda_max = 1500
width_g_max = 3
width_l_max = 3
lam1 = 588.966
lam2 = 589.562
obs = np.loadtxt("MICAlibs.txt")
x = obs[:,0]
sample_index = input("Input sample index (number of COLUMN):   ")
obs[:,1]= obs[:,int(sample_index)]
data1 = np.loadtxt("NIST/Na1.txt")
mask1 = (data1[:,0]>200) & (data1[:,0]<1000)
data = data1[mask1]
Stark_w = .5570e-5
l = data[:,0]
aki = data[:,1]
ei = data[:,2]
ek = data[:,3]
gi = data[:,4]
gk = data[:,5]
I =((aki*gk)/(l))*np.exp(-(ek)/(T*100))
peak,_ = scipy.signal.find_peaks(I, prominence=10000)
larray = [589.592424]
while True:
    class ZoomPan:
        def __init__(self):
            self.press = None
            self.cur_xlim = None
            self.cur_ylim = None
            self.x0 = None
            self.y0 = None
            self.x1 = None
            self.y1 = None
            self.xpress = None
            self.ypress = None
        def zoom_factory(self, ax1, base_scale = 2.):
            def zoom(event):
                cur_xlim = ax1.get_xlim()
                cur_ylim = ax1.get_ylim()
                xdata = event.xdata
                ydata = event.ydata
                if event.button == 'down':
                    scale_factor = 1 / base_scale
                elif event.button == 'up':
                    scale_factor = base_scale
                else:
                    scale_factor = 1
                    print(event.button)
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
                ax1.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
                ax1.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
                ax1.figure.canvas.draw()
            fig = ax1.get_figure() # get the figure of interest
            fig.canvas.mpl_connect('scroll_event', zoom)
            return zoom
        def pan_factory(self, ax1):
            def onPress(event):
                if event.inaxes != ax1: return
                self.cur_xlim = ax1.get_xlim()
                self.cur_ylim = ax1.get_ylim()
                self.press = self.x0, self.y0, event.xdata, event.ydata
                self.x0, self.y0, self.xpress, self.ypress = self.press
            def onRelease(event):
                self.press = None
                ax1.figure.canvas.draw()
            def onMotion(event):
                if self.press is None: return
                if event.inaxes != ax1: return
                dx = event.xdata - self.xpress
                dy = event.ydata - self.ypress
                self.cur_xlim -= dx
                self.cur_ylim -= dy
                ax1.set_xlim(self.cur_xlim)
                ax1.set_ylim(self.cur_ylim)
                ax1.figure.canvas.draw()
            fig1 = ax1.get_figure()
            fig1.canvas.mpl_connect('button_press_event',onPress)
            fig1.canvas.mpl_connect('button_release_event',onRelease)
            fig1.canvas.mpl_connect('motion_notify_event',onMotion)
            return onMotion
    fig1 = figure()
    #for lam in larray:
    for lam in larray:
        mask = (x>lam-1) & (x<lam+1)
        c_o = []
        b_l = []
        A_ki = aki[l == lam]
        area = 0
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(obs[mask][:,0],obs[mask][:,1], label = 'obs')
#        ax1.axvline(lam, label = 'Standerd Peak Position')
        ax1.set_ylabel(r'$I \rightarrow$')
        ax1.set_xlabel(r'$\lambda \longrightarrow$')
        ax1.set_title("Peak at $\lambda = $"+str(lam)+"  $A_{ki} = $"+str(A_ki))
        ax1.grid()
        ax1.legend(loc = 'upper left')
        scale = 1.1
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax1, base_scale = scale)
        figPan = zp.pan_factory(ax1)
        plt.title("Peak at $\lambda = $"+str(lam)+"  $A_{ki} = $"+str(A_ki))
        fig1.show()
        coords_wl = []
        coords_int = []
        co = []
        def onclick(event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            global coords_wl
            coords_wl.append(ix)
            global coords_int
            coords_int.append(iy)
            return coords_wl, coords_int
        cid = fig1.canvas.mpl_connect('button_press_event', onclick)
        while True:
            sele = input("Remove background" )
            if sele == 'n':
                break
            else:
                x0 = coords_wl[0]
                x1 = coords_wl[-1]
                x2 = coords_wl[1]
                x3 = coords_wl[2]
                x4 = coords_wl[3]
                x5 = coords_wl[4]
                x6 = coords_wl[5]
                x7 = coords_wl[6]
                y0 = coords_int[0]
                y1 = coords_int[-1]
                y2 = coords_int[1]
                y3 = coords_int[2]
                y4 = coords_int[3]
                y5 = coords_int[4]
                y6 = coords_int[5]
                y7 = coords_int[6]
                x_values = obs[mask][:,0]
                def background(x_values, x0,x1,y0,y1):
                    BG = (((y1-y0)/(x1-x0))*(x_values - x0)) + y0
                    return BG
                correct_out = obs[mask][:,0],obs[mask][:,1] - background(x_values, x0,x1,y0,y1)
                plt.plot(obs[mask][:,0],obs[mask][:,1], label = 'obs')
                #plt.plot(obs[mask][:,0],b_l, '--',  label = 'background')
                plt.axvline(lam, label = 'Standerd Peak Position')
                plt.legend(loc = 'upper left')
                mask_int = (x0 <= correct_out[0])&(correct_out[0] <= x1)
                x_int = correct_out[0][mask_int]
                y_int = correct_out[1][mask_int]
                for x1 in range(np.size(correct_out[0][mask_int]) - 1):
                    area1 = ((x_int[x1+1]-x_int[x1])*y_int[x1+1]) - ((1/2)*(x_int[x1+1]-x_int[x1])*(y_int[x1+1]-y_int[x1]))
                    area += area1
                #plt.fill_between(correct_out[0],correct_out[1], color = 'gray')
                plt.plot(x_int, y_int, label = 'corrected')
                plt.title("Peak at $\lambda = $"+str(lam)+"  $A_{ki} = $"+str(A_ki))
                plt.ylabel(r'$I \rightarrow$')
                plt.xlabel(r'$\lambda \longrightarrow$')
                plt.grid()
                plt.show()
                while True:
                    sel = input("Remove peak at "+str(lam)+ ":           [Input y]           " )
                    if sel == 'y':
                        print("This Line is removed")
                        break
                    else:
                        with open('Out/area'+str(lam)+'_'+str(sample_index)+'.txt', 'w') as f1:
                            f1.write(str(area))
                        with open('Out/correct_out'+str(lam+1)+'_'+str(sample_index)+'.txt', 'w') as f2:
                            f2.write(str(correct_out))
                        param_bounds=([0,0,0,0,0,0.0,0,0,0,0,0,0.0],[np.inf,np.inf,lambda_max,3,3,1,np.inf,np.inf,lambda_max,3,3,1])
                        p = 0.5
                        popt_pv = []
                        [amp_g, amp_l, cen, width_g, width_l, p] = [np.max(y_int),np.max(y_int), lam, x_int[-1]-x_int[0] , x_int[-1]-x_int[0], .5]
                        def pv1(x_int, amp_g1, amp_l1, cen1, width_g1, width_l1, p1):
                            return p1*((amp_g1/np.sqrt(2*np.pi*width_g1**2))*np.exp(-((x_int-cen1)**2)/(2*width_g1**2)))+(1-p1)*((amp_l1/np.pi)*((width_l1/2)**2/((x_int-cen1)**2+(width_l1/2)**2)))
                        def pv2(x_int, amp_g2, amp_l2, cen2, width_g2, width_l2, p2 ):
                            return p2*((amp_g2/np.sqrt(2*np.pi*width_g2**2))*np.exp(-((x_int-cen2)**2)/(2*width_g2**2)))+(1-p2)*((amp_l2/np.pi)*((width_l2/2)**2/((x_int-cen2)**2+(width_l2/2)**2)))
                        def Dpv(x_int, amp_g1, amp_l1, cen1, width_g1, width_l1, p1, amp_g2, amp_l2, cen2, width_g2, width_l2, p2):
                            return pv1(x_int, amp_g1, amp_l1, cen1, width_g1, width_l1, p1) + pv2(x_int, amp_g2, amp_l2, cen2, width_g2, width_l2, p2 )
                        popt_Dpv,_ = scipy.optimize.curve_fit(Dpv, x_int, y_int, p0 = [y3,y3, x3, x4-x2, x4-x2, .5, y6,y6, x6, x7-x5, x7-x5, .5] ,bounds = param_bounds, maxfev=5000000000)
                        amp_g1 = popt_Dpv[0]
                        amp_l1 = popt_Dpv[1]
                        cen1 = popt_Dpv[2]
                        width_g1 = popt_Dpv[3]
                        width_l1 = popt_Dpv[4]
                        p1 = popt_Dpv[5]
                        amp_g2 = popt_Dpv[6]
                        amp_l2 = popt_Dpv[7]
                        cen2 = popt_Dpv[8]
                        width_g2 = popt_Dpv[9]
                        width_l2 = popt_Dpv[10]
                        p2 = popt_Dpv[11]
                        def Gauss(x_int, amp_g, cen, width_g, p ):
                            return p*((amp_g/np.sqrt(2*np.pi*width_g**2))*np.exp(-((x_int-cen)**2)/(2*width_g**2)))
                        def Lorentz(x_int, amp_l, cen, width_l, p ):
                            return (1-p)*((amp_l/np.pi)*((width_l/2)**2/((x_int-cen)**2+(width_l/2)**2)))
                        x_int_smooth = np.arange(x_int[0],x_int[-1],0.0001)
                        DPV = Dpv(x_int_smooth, *popt_Dpv)
                        C = 4.700e+21
                        #FWHM_g=popt_pv[3]*np.sqrt(4*np.log(2))
                        #N_e = (C)/((popt_pv[2]/10)**2*(10**(-8))*(Ek-Ei))
                        gaussian1 = Gauss(x_int_smooth, amp_g1, cen1, width_g1, p1 )
                        gaussian2 = Gauss(x_int_smooth, amp_g2, cen2, width_g2, p2 )
                        lorentzian1 = Lorentz(x_int_smooth, amp_l1, cen1, width_l1, p1 )
                        lorentzian2 = Lorentz(x_int_smooth, amp_l2, cen2, width_l2, p2 )
                        #def T_e(FWHM_l):
                            #return np.exp((-a1+np.sqrt(np.abs((a1**2)-(4*a2*(a0-np.log(FWHM_l))))))/(2*a2))
                        fig, ax = plt.subplots()
                        ax.plot(x_int, y_int, label = 'Corrected observation')
                        lin1, = ax.plot(x_int_smooth, Gauss(x_int_smooth, amp_g1, cen1, width_g1, p1 ), label = 'Gaussian1 component')
                        lin2, = ax.plot(x_int_smooth, Gauss(x_int_smooth, amp_g2, cen2, width_g2, p2 ), label = 'Gaussian2 component')
                        lin3, = ax.plot(x_int_smooth, Lorentz(x_int_smooth, amp_l1, cen1, width_l1, p1 ), label = 'lorentzian1 component')
                        lin4, = ax.plot(x_int_smooth, Lorentz(x_int_smooth, amp_l2, cen2, width_l2, p2 ), label = 'lorentzian2 component')
                        lin5, = ax.plot(x_int_smooth, Dpv(x_int_smooth, popt_Dpv[0],popt_Dpv[1],popt_Dpv[2],popt_Dpv[3],popt_Dpv[4],popt_Dpv[5],popt_Dpv[6],popt_Dpv[7],popt_Dpv[8],popt_Dpv[9],popt_Dpv[10],popt_Dpv[11]), label = 'PV Fit')
                        #plt.text(eq1, {'color': 'C2', 'fontsize': 18}, va="top", ha="right")
                        #plt.title("Pesudo Voigt Fitting at $\lambda = $"+str(lam))
                        plt.xlabel(r'$\lambda \longrightarrow$')
                        plt.ylabel(r'$I \rightarrow$')
                        plt.legend(loc = 'upper right')
                        axp1 = plt.axes([0.25, 0.1, 0.65, 0.03])
                        plt.subplots_adjust(left=0.25, bottom=0.25)
                        p1_slider = Slider(
                        ax=axp1,
                        label='$p_1$',
                        valmin=0,
                        valmax=1,
                        valinit=p1,
                        )
                        plt.subplots_adjust(left=0.25, bottom=0.25)
                        axp2 = plt.axes([0.25, 0.13, 0.65, 0.03])
                        p2_slider = Slider(
                        ax=axp2,
                        label="$p_2$",
                        valmin=0,
                        valmax=1,
                        valinit=p2,
                        )
                        def update(val):
                            lin1.set_ydata( Gauss(x_int_smooth, amp_g1, cen1, width_g1, p1_slider.val ))
                            lin2.set_ydata( Gauss(x_int_smooth, amp_g2, cen2, width_g2, p2_slider.val ))
                            lin3.set_ydata( Lorentz(x_int_smooth, amp_l1, cen1, width_l1, p1_slider.val ))
                            lin4.set_ydata( Lorentz(x_int_smooth, amp_l2, cen2, width_l2, p2_slider.val ))
                            lin5.set_ydata( Dpv(x_int_smooth, popt_Dpv[0],popt_Dpv[1],popt_Dpv[2],popt_Dpv[3],popt_Dpv[4],p1_slider.val,popt_Dpv[6],popt_Dpv[7],popt_Dpv[8],popt_Dpv[9],popt_Dpv[10],p2_slider.val))
                            fig.canvas.draw_idle()
                        p1_slider.on_changed(update)
                        p2_slider.on_changed(update)
                        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
                        button = Button(resetax, 'Reset', hovercolor='0.975')
                        def reset(event):
                            cen1_slider.reset()
                            cen2_slider.reset()
                        button.on_clicked(reset)
                        plt.show()
