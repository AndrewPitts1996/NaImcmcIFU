
from astropy import units as u
from astropy.coordinates import Angle
from scipy import interpolate
from scipy import special
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from astropy.io import fits
from IPython import embed

# 0. figure out PA -- ok
# 1. try simple rotating disk, compare with warped disk -- ok
# 2. implement accreting layer; may just need to adjust values of xC and yC
# 2a. yes, first implement costheta in the frame of the ring (pretending z=0)
# 2b. then shift vLOS array over in X and Y according to Section 5 in inspector.pdf
#     (maybe this is just adding constants to data[0], data[1]!)


#############################################################
#          Function to assemble several disk models
##############################################################            
def Assemble_DiskColumnDensity(vInf, deltaV, phi0, phiInf, Rphi, deltaphi, theta0, thetaInf, Rtheta, deltatheta, 
                                   n0, h, Rscale, dz, zlim, outfil=None):

    vLag = 0.0
    vAcc = 0.0

    # Set up array of heights
    z_array = np.arange(0,zlim,dz)

    for qq in range(len(z_array)):
        rotdisk = disk_model(vInf, deltaV, phi0, phiInf, Rphi, deltaphi,
                         theta0, thetaInf, Rtheta, deltatheta,
                         zAcc=z_array[qq], vLag=0.0, vAcc=0.0,
                         n0=n0, h=h, Rscale=Rscale, dz=dz)
        rotdisk.Set_Rings(60.0, 0.1)
        rotdisk.Phi_r()
        rotdisk.Theta_r()
        rotdisk.Density_r()
        rotdisk.Rings_columndensity([-60.0, 60.0], 0.1, density_flg='columndensity')

        if qq==0:

            xx = rotdisk.dumxx
            yy = rotdisk.dumyy
            sum_SurfaceDensity = rotdisk.SurfaceDensity_interp 

        else:

            sum_SurfaceDensity = sum_SurfaceDensity + (2.0*rotdisk.SurfaceDensity_interp)

    #embed()
    if(outfil!=None):

        np.savez_compressed(outfil, xx=xx, yy=yy, SurfaceDensity=sum_SurfaceDensity)
        #c1 = fits.Column(name='xx', array=xx, format='E')
        #c2 = fits.Column(name='yy', array=yy, format='E')
        #c3 = fits.Column(name='SurfaceDensity', array=sum_SurfaceDensity, format='E')
        #t = fits.BinTableHDU.from_columns([c1,c2,c3])
        #t.writeto(outfits, overwrite=True)

    return {'xx':xx, 'yy':yy, 'SurfaceDensity':sum_SurfaceDensity}
            

class disk_model:

    def __init__(self, vInf, deltaV, phi0, phiInf, Rphi, deltaphi,
                     theta0, thetaInf, Rtheta, deltatheta, zAcc=None, vLag=None, vAcc=None,
                     n0=None, h=None, Rscale=None, dz=None, ## set these for n0 analysis
                     Sigma_SFR0=None, R50=None):   ## set these for Sigma_SFR analysis; need to also set dz

        ## Set up rotation curve, inclination (phi), PA of disk as a function of r
        
        #self.r = np.arange(0, 50, 0.01)
        #self.vRot_r = vInf * np.tanh(self.r / deltaV)    ## km/s
        #self.Phi_r = phi0 + 0.5*(phiInf - phi0)*(1.0 + np.tanh((self.r-Rphi)/deltaphi))               ## degrees
        #self.Theta_r = theta0 + 0.5*(thetaInf - theta0)*(1.0 + np.tanh((self.r-Rtheta)/deltatheta))   ## degrees
        
        self.vInf = vInf
        self.deltaV = deltaV
        self.phi0 = phi0
        self.phiInf = phiInf
        self.Rphi = Rphi
        self.deltaphi = deltaphi
        self.theta0 = theta0
        self.thetaInf = thetaInf
        self.Rtheta = Rtheta
        self.deltatheta = deltatheta
        self.xC = 0.0    # center of each ring -- could make this variable later
        self.yC = 0.0
        self.vSYS = 0.0

        # If extraplanar layer
        if zAcc:
            self.extr_flg = True
            self.zAcc = zAcc
            self.vLag = vLag
            self.vAcc = vAcc

        else:
            self.extr_flg = False

        # If assigning densities
        if(n0):
            self.n0 = n0
            self.h = h
            self.Rscale = Rscale
            self.dz = dz

         # If assigning Sigma_SFR
        if(Sigma_SFR0):
            self.Sigma_SFR0 = Sigma_SFR0
            self.R50 = R50
            self.dz = dz
            self.Tot_SFR = Sigma_SFR0 * (R50**2) * 2.0 * np.pi * special.gamma(2) / (1.678**2)

            
    def Set_Rings(self, maxR, dR):

        ## Generate rings with a maximum radius maxR, at intervals dR
        self.rings_r = np.arange(0, maxR, dR)

    def vRot_r(self):

        ## Need to set rings before doing this
        rings_r = self.rings_r
        vRot_r = []

        for qq in range(len(rings_r)):
            rings_vRot = self.vInf * np.tanh(rings_r[qq] / self.deltaV)    ## km/s
            vRot_r.append(rings_vRot)
        self.vRot_r = np.array(vRot_r)
        
    def Phi_r(self):

        ## Need to set rings before doing this
        rings_r = self.rings_r
        Phi_r = []
        
        for qq in range(len(rings_r)):
            rings_Phi = self.phi0 + 0.5*(self.phiInf - self.phi0)*(1.0 + np.tanh((rings_r[qq]-self.Rphi)/self.deltaphi))   ## degrees
            Phi_r.append(rings_Phi)
        self.Phi_r = np.array(Phi_r)

    def Theta_r(self):

        ## Need to set rings before doing this
        rings_r = self.rings_r
        Theta_r = []

        for qq in range(len(rings_r)):
            rings_Theta = self.theta0 + 0.5*(self.thetaInf - self.theta0)*(1.0 + np.tanh((rings_r[qq]-self.Rtheta)/self.deltatheta))    
            Theta_r.append(rings_Theta)
        self.Theta_r = np.array(Theta_r)

    def Density_r(self):

        rings_r = self.rings_r
        Density_r = []

        if self.extr_flg:
            zAcc = self.zAcc
        else:
            zAcc = 0.0

        
        Density_height = self.n0 * np.exp(-1.0 * np.abs(zAcc) / self.h)

        for qq in range(len(rings_r)):
            rings_Density = Density_height * np.exp(-1.0 * rings_r[qq] / self.Rscale)
            Density_r.append(rings_Density)

        self.Density_r = np.array(Density_r)


    def SFRVolDensity_r(self):

        rings_r = self.rings_r
        SFRVolDensity_r = []

        SFRVolDensity_0 = self.Sigma_SFR0 / self.dz

        for qq in range(len(rings_r)):
            rings_SFRVolDensity = SFRVolDensity_0 * np.exp(-1.678 * rings_r[qq] / self.R50)
            SFRVolDensity_r.append(rings_SFRVolDensity)

        self.SFRVolDensity_r = np.array(SFRVolDensity_r)
        

    def vRot_extraplanar(self):

        rings_r = self.rings_r
        vRot_extraplanar = []
        
        for qq in range(len(rings_r)):
            rings_vRot_extrap = (self.vInf + (self.vLag * self.zAcc))*np.tanh(rings_r[qq] / self.deltaV)    ## km/s
            vRot_extraplanar.append(rings_vRot_extrap)
        self.vRot_extraplanar = np.array(vRot_extraplanar)

        
    def Rings_vLOS(self, coordrange, pixsz, outfil=None):

        ## Calculates los velocity for all rings in model
        ## phi = disk PA
        ## incl = disk inclination
        ## coordrange: determines x, y arrays onto which velocities are interpolated
        ## pixsz: size of pixels in these x, y arrays

        rings_r = self.rings_r
        Phi_r = self.Phi_r
        Theta_r = self.Theta_r
        xC = self.xC
        yC = self.yC
        vSYS = self.vSYS
        vEXP = 0.0

        if self.extr_flg:
            #print("Working with an extraplanar layer")
            
            zAcc = self.zAcc
            vLag = self.vLag
            vAcc = self.vAcc
            vRot_extraplanar = self.vRot_extraplanar

        else:
            vRot_r = self.vRot_r
        
        incl = Angle(Phi_r, 'deg')
        pa = Angle(Theta_r, 'deg')   ## ADD 90deg TO THIS
        alpha_pa = Angle(Theta_r+90.0, 'deg')
        #pdb.set_trace()
        rings_minor = np.cos(incl.radian) * rings_r
        
        ## Setup for plotting velocity field
        dumx = np.arange(coordrange[0], coordrange[1], pixsz)
        dumy = np.arange(coordrange[0], coordrange[1], pixsz)
        dumxx, dumyy = np.meshgrid(dumx, dumy)
        vLOS_arr = np.zeros((len(dumx),len(dumy)))
  
        
        ## Copied following ellipse code from
        ## https://casper.berkeley.edu/astrobaki/index.php/Plotting_Ellipses_in_Python
        thth = np.linspace(0.0, 2.0*np.pi, num=1000)
        rr = 1.0 / np.sqrt((np.cos(thth))**2 + (np.sin(thth))**2)
        xx = rr * np.cos(thth)
        yy = rr * np.sin(thth)
        data_fid = np.array([xx,yy])

        if self.extr_flg:
            zz = np.zeros(len(thth)) + zAcc
        
        
        
        for qq in range(len(rings_r)):

            X_arr = rings_r[qq] * ((xx*np.cos(alpha_pa.radian[qq])) - (yy*np.sin(alpha_pa.radian[qq])*np.cos(incl.radian[qq])))
            Y_arr = rings_r[qq] * ((xx*np.sin(alpha_pa.radian[qq])) + (yy*np.cos(alpha_pa.radian[qq])*np.cos(incl.radian[qq])))
            Z_arr = rings_r[qq] * yy * np.sin(incl.radian[qq])
            data = np.array([X_arr, Y_arr])
            
            #pdb.set_trace()
            
            ## Now have x, y arrays that trace ring:
            data[0] += xC
            data[1] += yC
            ## end astrobaki ellipse code

            ## From Oh, Staveley-Smith, Spekkens et al.
            dist_el = ((data[0] - xC)**2 + (data[1] - yC)**2)**0.5
            cos_theta = ((-1.0*(data[0] - xC)*np.sin(pa.radian[qq])) + ((data[1] - yC)*np.cos(pa.radian[qq]))) / dist_el
            sin_theta = ((-1.0*(data[0] - xC)*np.cos(pa.radian[qq])) - ((data[1] - yC)*np.sin(pa.radian[qq]))) / (dist_el*np.cos(incl.radian[qq]))

            if self.extr_flg:
                vLOS = vSYS + (np.sin(incl.radian[qq]) * (vRot_extraplanar[qq]*cos_theta)) + (vAcc * np.cos(incl.radian[qq]))

                ## Compute x, y offset
                Xarr_zterm = zz * np.sin(alpha_pa.radian[qq]) * np.sin(incl.radian[qq])
                Yarr_zterm = -1.0 * zz * np.cos(alpha_pa.radian[qq]) * np.sin(incl.radian[qq])
                Zarr_zterm = zz * np.cos(incl.radian[qq])

                data[0] += Xarr_zterm
                data[1] += Yarr_zterm
                Z_arr += Zarr_zterm
                #print("Adding to x, y, z for extraplanar layer:", Xarr_zterm[0], Yarr_zterm[0], Zarr_zterm[0])
                
            else:
                vLOS = vSYS + (np.sin(incl.radian[qq]) * (vRot_r[qq]*cos_theta + vEXP*sin_theta))

            ## Step through all points along ellipse
            #print("Filling in velocity field for ring ", qq)
            #for ii in range(len(data[0])):
            #    min_arr = np.abs(dumxx-data[0][ii]) + np.abs(dumyy-data[1][ii])
            #    idx = np.unravel_index(np.argmin(min_arr, axis=None), min_arr.shape)
            #    vLOS_arr[idx] = vLOS[ii]


            ## Concatenate x values, y values, vLOS values
            if(qq==0):
                x_all = data[0]
                y_all = data[1]
                z_all = Z_arr
                vLOS_all = vLOS

                ring_x = data[0]
                ring_y = data[1]
                #ring_z = Z_arr
            else:
                x_all = np.concatenate((x_all, data[0]))
                y_all = np.concatenate((y_all, data[1]))
                z_all = np.concatenate((z_all, Z_arr))
                vLOS_all = np.concatenate((vLOS_all, vLOS))

                ring_x = np.vstack((ring_x, data[0]))
                ring_y = np.vstack((ring_y, data[1]))
                #ring_z = np.vstack((ring_z, Z_arr))

        self.ring_x = ring_x
        self.ring_y = ring_y
        
        vLOS_interp = interpolate.griddata((x_all, y_all), vLOS_all, (dumxx, dumyy), method='linear')
        print(x_all.shape)
        print(np.shape(vLOS_all))
        z_interp = interpolate.griddata((x_all, y_all), z_all, (dumxx, dumyy), method='linear')
        self.vLOS_interp = vLOS_interp
        self.z_interp = z_interp
        self.dumxx = dumxx
        self.dumyy = dumyy

        if(outfil!=None):
            fig, ax = plt.subplots(1, 1)
            cax = ax.imshow(vLOS_interp, interpolation='nearest', cmap='coolwarm', origin='lower',
                            extent=[coordrange[0], coordrange[1], coordrange[0], coordrange[1]], vmin=-200.0, vmax=200.0)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            #pdb.set_trace()
            for qq in range(len(rings_r)):
                ax.plot(self.ring_x[qq,:], self.ring_y[qq,:], color='black')
        
            cbar = fig.colorbar(cax)
            cbar.set_label("Velocity (km/s)")
            plt.savefig(outfil, format='pdf')
          
        


    def Rings_columndensity_or_SigmaSFR(self, coordrange, pixsz, density_flg='SigmaSFR', outfil=None):

        ## Calculates los column density for all rings in model
        ## phi = disk PA
        ## incl = disk inclination
        ## coordrange: determines x, y arrays onto which velocities are interpolated
        ## pixsz: size of pixels in these x, y arrays

        if density_flg=='columndensity':
            Density_r = self.Density_r
            
        elif density_flg=='SigmaSFR':
            Density_r = self.SFRVolDensity_r

        else:
            print("Must set type of surface density to calculate")
            embed()

        
        
        rings_r = self.rings_r
        Phi_r = self.Phi_r
        Theta_r = self.Theta_r
        #Density_r = self.Density_r
        xC = self.xC
        yC = self.yC
        dz = self.dz   # Thickness of layer in kpc
        
        #vSYS = self.vSYS
        #vEXP = 0.0

        if self.extr_flg:
            zAcc = self.zAcc
        else:
            zAcc = 0.0
        
        incl = Angle(Phi_r, 'deg')
        pa = Angle(Theta_r, 'deg')   ## ADD 90deg TO THIS
        alpha_pa = Angle(Theta_r+90.0, 'deg')
        #pdb.set_trace()
        rings_minor = np.cos(incl.radian) * rings_r
        
        ## Setup for plotting velocity field
        dumx = np.arange(coordrange[0], coordrange[1], pixsz)
        dumy = np.arange(coordrange[0], coordrange[1], pixsz)
        dumxx, dumyy = np.meshgrid(dumx, dumy)
        columndensity_arr = np.zeros((len(dumx),len(dumy)))
  
        
        ## Copied following ellipse code from
        ## https://casper.berkeley.edu/astrobaki/index.php/Plotting_Ellipses_in_Python
        thth = np.linspace(0.0, 2.0*np.pi, num=1000)
        rr = 1.0 / np.sqrt((np.cos(thth))**2 + (np.sin(thth))**2)
        xx = rr * np.cos(thth)
        yy = rr * np.sin(thth)
        data_fid = np.array([xx,yy])

        if self.extr_flg:
            zz = np.zeros(len(thth)) + zAcc
        else:
            zz = np.zeros(len(thth))
        
        for qq in range(len(rings_r)):

            X_arr = rings_r[qq] * ((xx*np.cos(alpha_pa.radian[qq])) - (yy*np.sin(alpha_pa.radian[qq])*np.cos(incl.radian[qq])))
            Y_arr = rings_r[qq] * ((xx*np.sin(alpha_pa.radian[qq])) + (yy*np.cos(alpha_pa.radian[qq])*np.cos(incl.radian[qq])))
            Z_arr = rings_r[qq] * yy * np.sin(incl.radian[qq])
            data = np.array([X_arr, Y_arr])
            
            ## Now have x, y arrays that trace ring:
            data[0] += xC
            data[1] += yC
            ## end astrobaki ellipse code

            ## From Oh, Staveley-Smith, Spekkens et al.
            dist_el = ((data[0] - xC)**2 + (data[1] - yC)**2)**0.5
            cos_theta = ((-1.0*(data[0] - xC)*np.sin(pa.radian[qq])) + ((data[1] - yC)*np.cos(pa.radian[qq]))) / dist_el
            sin_theta = ((-1.0*(data[0] - xC)*np.cos(pa.radian[qq])) - ((data[1] - yC)*np.sin(pa.radian[qq]))) / (dist_el*np.cos(incl.radian[qq]))

            if density_flg=='columndensity':
                SurfaceDensity = (Density_r[qq] * (dz * u.kpc.to(u.cm) / np.cos(incl.radian[qq]))) + np.zeros(len(cos_theta))
            elif density_flg=='SigmaSFR':
                SurfaceDensity = (Density_r[qq] * (dz / np.cos(incl.radian[qq]))) + np.zeros(len(cos_theta))
            
            if self.extr_flg:
                #vLOS = vSYS + (np.sin(incl.radian[qq]) * (vRot_extraplanar[qq]*cos_theta)) + (vAcc * np.cos(incl.radian[qq]))

                ## Compute x, y offset
                Xarr_zterm = zz * np.sin(alpha_pa.radian[qq]) * np.sin(incl.radian[qq])
                Yarr_zterm = -1.0 * zz * np.cos(alpha_pa.radian[qq]) * np.sin(incl.radian[qq])
                Zarr_zterm = zz * np.cos(incl.radian[qq])

                data[0] += Xarr_zterm
                data[1] += Yarr_zterm
                Z_arr += Zarr_zterm
                #print("Adding to x, y, z for extraplanar layer:", Xarr_zterm[0], Yarr_zterm[0], Zarr_zterm[0])
                
            #else:
            #vLOS = vSYS + (np.sin(incl.radian[qq]) * (vRot_r[qq]*cos_theta + vEXP*sin_theta))


            ## Step through all points along ellipse
            #print("Filling in velocity field for ring ", qq)
            #for ii in range(len(data[0])):
            #    min_arr = np.abs(dumxx-data[0][ii]) + np.abs(dumyy-data[1][ii])
            #    idx = np.unravel_index(np.argmin(min_arr, axis=None), min_arr.shape)
            #    vLOS_arr[idx] = vLOS[ii]


            ## Concatenate x values, y values, vLOS values
            
            if(qq==0):
                x_all = data[0]
                y_all = data[1]
                z_all = Z_arr
                SurfaceDensity_all = SurfaceDensity

                ring_x = data[0]
                ring_y = data[1]
                #ring_z = Z_arr
            else:
                x_all = np.concatenate((x_all, data[0]))
                y_all = np.concatenate((y_all, data[1]))
                z_all = np.concatenate((z_all, Z_arr))
                
                SurfaceDensity_all = np.concatenate((SurfaceDensity_all, SurfaceDensity))

                ring_x = np.vstack((ring_x, data[0]))
                ring_y = np.vstack((ring_y, data[1]))
                #ring_z = np.vstack((ring_z, Z_arr))

        self.ring_x = ring_x
        self.ring_y = ring_y

        SurfaceDensity_interp = interpolate.griddata((x_all, y_all), SurfaceDensity_all, (dumxx, dumyy), method='linear')
        z_interp = interpolate.griddata((x_all, y_all), z_all, (dumxx, dumyy), method='linear')
        self.SurfaceDensity_interp = SurfaceDensity_interp
        self.z_interp = z_interp
        self.dumxx = dumxx
        self.dumyy = dumyy
        
        if(outfil!=None):
            fig, ax = plt.subplots(1, 1)
            cmap = matplotlib.cm.jet
            cmap.set_bad(color='gray')
            
            cax = ax.imshow(SurfaceDensity_interp, cmap=cmap, origin='lower',
                            extent=[coordrange[0], coordrange[1], coordrange[0], coordrange[1]], norm=LogNorm(vmin=2.3e-5, vmax=0.1))
            #cax = ax.imshow(SurfaceDensity_all, interpolation='nearest', cmap='coolwarm', origin='lower')
            
            #for qq in range(len(rings_r)):
            #    ax.plot(self.ring_x[qq,:], self.ring_y[qq,:], color='black')
        
            cbar = fig.colorbar(cax)
            cbar.set_label("Surface Density")
            plt.savefig(outfil, format='pdf')
          
            

        
    ############################################
    #                PLOTTING
    ############################################
    def Plot_Rings(self, xrange, yrange, outfil):

        # First calculate minor axis length from inclination
        rings_r = self.rings_r
        incl = Angle(self.Phi_r, 'deg')
        pa = Angle(self.Theta_r, 'deg')

        rings_minor = np.cos(incl.radian) * rings_r

        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        for qq in range(len(rings_r)):
            #qq = 5
            ell = mpatches.Ellipse((0.0,0.0), 2.0*rings_r[qq], 2.0*rings_minor[qq], (pa.degree[qq] + 90.0),
                                    edgecolor='red', facecolor='none')
            ax.add_patch(ell)
        plt.savefig(outfil, format='pdf')
                                   
        

    


