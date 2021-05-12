
import matplotlib.pyplot as plt
import numpy as np

class TE():


    def Ex(x, y, z, m, n, a, b, h, K ,A, w, t):
        return ((-A*w*((np.pi * n) / b) /K**2) * np.cos((np.pi * m * x) / a) * np.sin((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Ey(x, y, z, m, n, a, b, h, K, A, w, t):
        return ((A*w*((np.pi * m) / a) /K**2) * np.sin((np.pi * m * x) / a) * np.cos((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Ez(x, y, z, m, n, a, b, h, K, A, w, t):
        return 0
    
    def Hx(x, y, z, m, n, a, b, h, K, A, w, t):
        return ((-A*h*((np.pi * m) / a) /K**2) * np.sin((np.pi * m * x) / a) * np.cos((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Hy(x, y, z, m, n, a, b, h, K, A, w, t):
        return ((-A*h*((np.pi * n) / b) /K**2) * np.cos((np.pi * m * x) / a) * np.sin((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Hz(x, y, z, m, n, a, b, h, K, A, w, t):
        return (A * np.cos((np.pi * m * x) / a) * np.cos((np.pi * n * y) / b)*np.cos(w*t-h*z))



class TM():
    def Ex(x, y, z, m, n, a, b, h, K ,A, w, t):
        return ((A*h*((np.pi * m) / a) /K**2) * np.cos((np.pi * m * x) / a) * np.sin((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Ey(x, y, z, m, n, a, b, h, K, A, w, t):
        return ((A*h*((np.pi * n) / b) /K**2) * np.sin((np.pi * m * x) / a) * np.cos((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Ez(x, y, z, m, n, a, b, h, K, A, w, t):
        return (A * np.sin((np.pi * m * x) / a) * np.sin((np.pi * n * y) / b)*np.cos(w*t-h*z))
    
    def Hx(x, y, z, m, n, a, b, h, K, A, w, t):
        return ((-A*w*((np.pi * n) / b) /K**2) * np.sin((np.pi * m * x) / a) * np.cos((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Hy(x, y, z, m, n, a, b, h, K, A, w, t):
        return ((A*w*((np.pi * m) / a) /K**2) * np.cos((np.pi * m * x) / a) * np.sin((np.pi * n * y) / b)*np.sin(w*t-h*z))

    def Hz(x, y, z, m, n, a, b, h, K, A, w, t):
        return 0


t=1
A=1#amplituda
m=1# мода
n=0#мода
a=10.#*np.pi#размер волновода по х
b=5.#*np.pi# размер волновода по у
c= 3*10**8 #скорость света
w=200000000  #частота
e=1 #эпсилон ε диэлектрическая проницаемость среды
u=1 # ню μ магнитная проницаемость среды



import matplotlib.gridspec as gridspec
from PyQt5 import QtWidgets
from window import Ui_MainWindow
from err import Ui_MainWindow as err
import sys
import matplotlib.animation as animation

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super (mywindow,self).__init__()
        self.ui= Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def btnClicked(self):
        try:
            t = float(self.ui.lineEdit_4.text()) #время
            m = int(self.ui.spinBox.value())
            n = int(self.ui.spinBox_2.value())
            a = float(self.ui.lineEdit.text())
            b = float(self.ui.lineEdit_2.text())
            w = float(self.ui.lineEdit_3.text())
            C = 1/int(self.ui.spinBox_3.value()) #кэф отрисовки
            K = np.sqrt((m*np.pi/a)**2+(n*np.pi/b)**2)# χ каппа поперечное волновое число 
            k = w/c*np.sqrt(e*u)# волновое число
            h = np.sqrt(k**2-K**2)# продольное волновое число
            Z0 = np.sqrt(u/e)#волновой импеданс
            ZTE = Z0*k/h#
            ZTM = Z0*h/k#
            wkr = K*c #критическая частота
            l = 2*np.pi/k#длина волны в свободном пространстве
            lkr = 2*np.pi/K#длина волны критическая
            lv = 2*np.pi/h#длина волны в волноводе
            vf = w/h#фазовая скорость
            vgp = c**2/vf#групповая скорость

            self.ui.label_11.setText("Критическая частота="+str(wkr))
            self.ui.label_12.setText("Длина волны в волноводе ="+str(lv))
            x=np.arange(0., a+C, C)      #
            y=np.arange(0., b+C, C)
            z = np.arange(0., a*2+C, C)
            x, y, z = np.meshgrid(x, y, z)
            if w==0 or w<wkr:
                    raise Exception("Частота должна быть больше критической частоты:{}Гц".format(str(round(wkr,3))))
            #fig1, ax8= plt.subplots()
                        

            if str(self.ui.comboBox.currentText())=="ТЕ":
                
                if m==0 and n==0:
                    raise Exception("m не может быть равным 0  одновременно с n равным 0")
                

                
                
                # ims=[]
                # for t in range(0,300,5):
                #     YZEx=TE.Ex(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                #     YZEy=TE.Ey(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                #     YZEz=0*YZEy
                #     lwyze=(YZEy**2+YZEz**2)**0.5/(YZEy.max()**2+YZEz.max()**2)**0.5
                #     #ax2=fig.add_subplot(gs[1,0])
                #     YZE=ax8.streamplot(z[:,1,:],y[:,1,:],YZEz,YZEy,linewidth=lwyze)
                #     ims.append([YZE])


                fig= plt.figure(figsize=(8,6))
                gs= gridspec.GridSpec(nrows=3,ncols=2,wspace = 0.4 , hspace = 0.4)
                fig.suptitle('TE волна моды {}{} '.format(m,n), fontsize=12)
                #YZ  
                #E
                YZEx=TE.Ex(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZEy=TE.Ey(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZEz=0*YZEy
                ax2=fig.add_subplot(gs[1,0])
                ax2.set_ylim(0,b+C)
                #lwyze=(YZEy**2+YZEz**2)**0.5/(abs(YZEy).max()**2+abs(YZEz).max()**2)**0.5
                
                
                #YZE=ax2.streamplot(z[:,1,:],y[:,1,:],YZEz,YZEy,linewidth=lwyze)
                

                if abs(YZEx).max()==0 and abs(YZEy).max()!=0 :
                    lwyze=(YZEy**2+YZEz**2)**0.5/(abs(YZEy).max()**2+abs(YZEz).max()**2)**0.5
                    
                    YZE=ax2.streamplot(z[:,1,:],y[:,1,:],YZEz,YZEy,linewidth=lwyze)

                elif abs(YZEy).max()==0 :

                    for i in range(0,len(y[:,0,0])-1,4):
                        for j in range(0,len(z[0,0,:]),5) :
                            if TE.Ex(x=x[0,1,0], y=y[i,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)>0:
                                mark="o"
                            else:
                                mark="x"
                            YZE=ax2.scatter(z[0,0,j],y[i,0,0],s=40*(abs(TE.Ex(x=x[0,1,0], y=y[i,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t))/abs(YZEx).max()),c="r", marker=mark)

                else:
                    for i in range(0,len(y[:,0,0])-1,4):
                        for j in range(0,len(z[0,0,:]),5) :
                            if TE.Ex(x=x[0,1,0], y=y[i,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)>0:
                                mark="o"
                            else:
                                mark="x"
                            YZE=ax2.scatter(z[0,0,j],y[i,0,0],s=40*(abs(TE.Ex(x=x[0,1,0], y=y[i,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t))/abs(YZEx).max()),c="r", marker=mark)

                    lwyze=(YZEy**2+YZEz**2)**0.5/(abs(YZEy).max()**2+abs(YZEz).max()**2)**0.5
                    YZE=ax2.streamplot(z[:,1,:],y[:,1,:],YZEz,YZEy,linewidth=lwyze)
                    

                ax2.set_xlabel('Z')
                ax2.set_ylabel('У')

                #H
                YZHx=TE.Hx(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZHy=TE.Hy(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZHz=TE.Hz(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)

                #XY
                #E
                XYEx=TE.Ex(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XYEy=TE.Ey(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XYEz=[0 for i in range(len(XYEy))]
                #H
                XYHx=TE.Hx(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XYHy=TE.Hy(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)

                #XZ
                #E

                XZEx=TE.Ex(x=x[1,:,:], y=y[1,0,0], z=z[1,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZEy=TE.Ey(x=x[1,:,:], y=y[1,0,0], z=z[1,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                #print(XZEx)
                XZEz=0*XZEx
                ax4=fig.add_subplot(gs[2,0])
                ax4.set_ylim(0,a+C)
                if abs(XZEy).max()==0 and abs(XZEx).max()!=0 :
                    lwxze=(XZEx**2+XZEz**2)**0.5/(abs(XZEx).max()**2+abs(XZEz).max()**2)**0.5
                    
                    XZE=ax4.streamplot(z[0,:,:],x[0,:,:],XZEz,XZEx,linewidth=lwxze)

                elif abs(XZEx).max()==0 :

                    for i in range(0,len(x[0,:,0]),5):
                        for j in range(0,len(z[0,0,:]),5) :
                            if TE.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)>0:
                                mark="o"
                            else:
                                mark="x"
                            XZE=ax4.scatter(z[0,0,j],x[0,i,0],s=40*(abs(TE.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t))/abs(XZEy).max()),c="r", marker=mark)

                else:
                    for i in range(0,len(x[0,:,0]),5):
                        for j in range(0,len(z[0,0,:]),5) :
                            if TE.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)>0:
                                mark="o"
                            else:
                                mark="x"
                            XZE=ax4.scatter(z[0,0,j],x[0,i,0],s=40*(abs(TE.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t))/abs(XZEy).max()),c="r", marker=mark)#, label='Ey')

                    lwxze=(XZEx**2+XZEz**2)**0.5/(abs(XZEx).max()**2+abs(XZEz).max()**2)**0.5
                    
                    XZE1=ax4.streamplot(z[0,:,:],x[0,:,:],XZEz,XZEx,linewidth=lwxze)

                ax4.set_xlabel('Z')
                ax4.set_ylabel('X')
                #ax4.legend([XZE,XZE1],['Ey','sqrt(Ex^2+Ez^2)'] )
                #H
                XZHx=TE.Hx(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZHy=TE.Hy(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZHz=TE.Hz(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)



            elif self.ui.comboBox.currentText() =="ТМ": #TM
                
                if m==0 or n==0:
                    raise Exception("m и n не могут быть равными 0  при ТМ волне")
                
                # ims=[]
                # for t in range(0,300,5):
                #     lwyze=((TM.Ey(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)**2+YZEz**2)**0.5/\
                #         (TM.Ey(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t).max()**2+\
                #         TM.Ez(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t).max()**2)**0.5)
                #     ax2=fig.add_subplot(gs[1,0])
                #     YZE=ax2.streamplot(z[:,1,:],y[:,1,:],TM.Ez(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t),\
                #         TM.Ey(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t),linewidth=lwyze)
                #     ims.append([YZE])
                fig= plt.figure(figsize=(8,6))
                gs= gridspec.GridSpec(nrows=3,ncols=2)
                fig.suptitle('TM2 волна моды {}{} '.format(m,n), fontsize=12)
                #YZ  
                # #E
                YZEx=TM.Ex(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZEy=TM.Ey(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZEz=TM.Ez(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                
                lwyze=(YZEy**2+YZEz**2)**0.5/(abs(YZEy).max()**2+abs(YZEz).max()**2)**0.5
                ax2=fig.add_subplot(gs[1,0])
                YZE=ax2.streamplot(z[:,1,:],y[:,1,:],YZEz,YZEy,linewidth=lwyze)
                ax2.set_xlabel('Z')
                ax2.set_ylabel('У')

                #H
                YZHx=TM.Hx(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZHy=TM.Hy(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                YZHz=0*YZHx  #TE.Hz(x=x[0][1], y=y[:,1,:], z=z[:,1,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)

                #XY
                #E
                XYEx=TM.Ex(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XYEy=TM.Ey(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XYEz=TM.Ez(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                #H
                XYHx=TM.Hx(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XYHy=TM.Hy(x=x[:,:,0], y=y[:,:,0], z=z[0][0][1], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)

                #XZ
                #E


                XZEx=TM.Ex(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZEy=TM.Ey(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZEz=TM.Ez(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)

                ax4=fig.add_subplot(gs[2,0])
                ax4.set_ylim(0,a+C+0.1) 
                if abs(XZEy).max()==0 and abs(XZEx).max()!=0 :
                    lwxze=(XZEx**2+XZEz**2)**0.5/(abs(XZEx).max()**2+abs(XZEz).max()**2)**0.5
                    
                    XZE=ax4.streamplot(z[0,:,:],x[0,:,:],XZEz,XZEx,linewidth=lwxze)

                elif abs(XZEx).max()==0 and abs(XZEz).max()==0:

                    for i in range(0,len(x[0,:,0]),5):
                        for j in range(0,len(z[0,0,:]),5) :
                            if TM.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)>0:
                                mark="o"
                            else:
                                mark="x"
                            XZE=ax4.scatter(z[0,0,j],x[0,i,0],s=40*(abs(TM.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t))/abs(XZEy).max()),c="r", marker=mark)

                else:
                    for i in range(0,len(x[0,:,0]),5):
                        for j in range(0,len(z[0,0,:]),5) :
                            if TM.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)>0:
                                mark="o"
                            else:
                                mark="x"
                            XZE=ax4.scatter(z[0,0,j],x[0,i,0],s=40*(abs(TM.Ey(x=x[0,i,0], y=y[0,0,0], z=z[0,0,j], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t))/abs(XZEy).max()),c="r", marker=mark, label='Ey')

                    lwxze=(XZEx**2+XZEz**2)**0.5/(abs(XZEx).max()**2+abs(XZEz).max()**2)**0.5
                    
                    XZE=ax4.streamplot(z[0,:,:],x[0,:,:],XZEz,XZEx,linewidth=lwxze)
                ax4.set_xlabel('Z')
                ax4.set_ylabel('X')
                #H
                XZHx=TM.Hx(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZHy=TM.Hy(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                XZHz=0*XZHx  #TE.Hz(x=x[0,:,:], y=y[0,0,0], z=z[0,:,:], m=m, n=n, a=a, b=b, h=h, K=K, A=A, w=w, t=t)
                
                
            
            # fig= plt.figure()
            # gs= gridspec.GridSpec(nrows=3,ncols=2)

            #XY
            lwxye=(XYEx**2+XYEy**2)**0.5/(abs(XYEx).max()**2+abs(XYEy).max()**2)**0.5
            ax0=fig.add_subplot(gs[0,0])
            XYE=ax0.streamplot(x[:,:,0],y[:,:,0],XYEx,XYEy,linewidth=lwxye)
            ax0.set_title('электрическое поле ')
            ax0.set_xlabel('Х')
            ax0.set_ylabel('У')
            lwxyh=(XYHx**2+XYHy**2)**0.5/(abs(XYHx).max()**2+abs(XYHy).max()**2)**0.5
            ax1=fig.add_subplot(gs[0,1])
            XYH=ax1.streamplot(x[:,:,0],y[:,:,0],XYHx,XYHy,linewidth=lwxyh)
            ax1.set_title('магнитное поле' )
            ax1.set_xlabel('Х')
            ax1.set_ylabel('У')
            #YZ

            #ani= animation.ArtistAnimation(fig1, ims,interval=200, blit=True, repeat_delay=1000)
            #writer=animation.FFMpegWriter(fps=15,bitrate=1800)
            #ani.save("graf.mp4", writer=writer)

            # lwyze=(YZEy**2+YZEz**2)**0.5/(abs(YZEy).max()**2+abs(YZEz).max()**2)**0.5
            # ax2=fig.add_subplot(gs[1,0])
            # YZE=ax2.streamplot(z[:,1,:],y[:,1,:],YZEz,YZEy,linewidth=lwyze)
            # ax2.set_xlabel('Z')
            # ax2.set_ylabel('У')
            lwyzh=(YZHy**2+YZHz**2)**0.5/(abs(YZHy).max()**2+abs(YZHz).max()**2)**0.5 
            ax3=fig.add_subplot(gs[1,1])
            YZH=ax3.streamplot(z[:,1,:],y[:,1,:],YZHz,YZHy,linewidth=lwyzh)
            ax3.set_xlabel('Z')
            ax3.set_ylabel('У')
            # #XZ
            # lwxze=(XZEx**2+XZEz**2)**0.5/(abs(XZEx).max()**2+abs(XZEz).max()**2)**0.5
            # ax4=fig.add_subplot(gs[2,0]) 
            # XZE=ax4
            # XZE=ax4.streamplot(z[0,:,:],x[0,:,:],XZEz,XZEx,linewidth=lwxze)

            lwxzh=(XZHx**2+XZHz**2)**0.5/(abs(XZHx).max()**2+abs(XZHz).max()**2)**0.5
            ax5=fig.add_subplot(gs[2,1])
            XZH=ax5.streamplot(z[0,:,:],x[0,:,:],XZHz,XZHx,linewidth=lwxzh)
            ax5.set_xlabel('Z')
            ax5.set_ylabel('X')


            plt.show()

        except Exception as ex:
            #print(ex)
            showeror.ui.label.setText("Введены не правильные данные\n"+str(ex))
            showeror.show()
            

class Errorr(QtWidgets.QMainWindow):
    def __init__(self):
        super(Errorr,self).__init__()
        self.ui=err()
        self.ui.setupUi(self)


app=QtWidgets.QApplication([])
aplication= mywindow()
aplication.show()
showeror=Errorr()
sys.exit(app.exec())
