import numpy as np
import math
import matplotlib.pyplot as plt
import pdb
from tiltedrings_losv import disk_model

## Eigenbrot -- SALT data, email Matt?
## Maybe just do a lag without a warp first!


## M33 Model
## From Corbelli & Schneider 1997
vInf = 236 # max rotation velocity (from Matthewson & Ford 1996)
deltaV = 8.47 # guess
# inclination angle
phi0 = 47 # from MAD survey
phiInf = 56.8 # guess
Rphi = 23.4
deltaphi = 3.37
# position angle PA
theta0 = 27 # from MAD survey
thetaInf = -13.0
Rtheta = 40.8
deltatheta = 7.00

m33 = disk_model(vInf, deltaV, phi0, phiInf, Rphi, deltaphi,
                     theta0, thetaInf, Rtheta, deltatheta, zAcc=None, vLag=None, vAcc=None)
m33.Set_Rings(60.0, 3.0)
m33.Phi_r()
m33.Theta_r()
m33.vRot_r()
m33.Rings_vLOS([-60.0, 60.0], 0.5, "m33_rings_losvel.pdf")
#m33.Plot_Rings([-60.0, 60.0], [-60.0, 60.0], "m33_tiltedrings.pdf")


## Extraplaner layer with M33-like tilted rings
zAcc = 2.0   # height of layer
vLag = -15.0   # halo lag in km/s/kpc
vAcc = 0.0   # accretion velocity in km/s
m33_extraplanar = disk_model(vInf, deltaV, phi0, phiInf, Rphi, deltaphi,
                                 theta0, thetaInf, Rtheta, deltatheta, zAcc=zAcc, vLag=vLag, vAcc=vAcc)


m33_extraplanar.Set_Rings(60.0, 3.0)
m33_extraplanar.Phi_r()
m33_extraplanar.Theta_r()
m33_extraplanar.vRot_extraplanar()
m33_extraplanar.Rings_vLOS([-60.0, 60.0], 0.5, "m33extraplanar_rings_losvel.pdf")




## For a simple rotating disk
phi0 = 49.0
phiInf = 49.0
theta0 = -13.0
thetaInf = -13.0

rotdisk = disk_model(vInf, deltaV, phi0, phiInf, Rphi, deltaphi,
                     theta0, thetaInf, Rtheta, deltatheta, zAcc=None, vLag=None, vAcc=None)

rotdisk.Set_Rings(60.0, 3.0)
rotdisk.Phi_r()
rotdisk.Theta_r()
rotdisk.vRot_r()
rotdisk.Rings_vLOS([-60.0, 60.0], 0.5, "rotdisk_rings_losvel.pdf")


#pdb.set_trace()


fig, ax = plt.subplots(1, 1)

whbg = np.where(m33.z_interp < rotdisk.z_interp)
diff_vLOS = m33.vLOS_interp - rotdisk.vLOS_interp
diff_vLOS[whbg] = 0.0
                
cax = ax.imshow((m33_extraplanar.vLOS_interp - rotdisk.vLOS_interp), interpolation='nearest', cmap='coolwarm', origin='lower',
                    extent=[-60,60,-60,60], vmin=-50.0, vmax=50.0)

cbar = fig.colorbar(cax)
cbar.set_label("Velocity (km/s)")
plt.savefig("diff_m33extra_rotdisk.pdf", format='pdf')

fig, ax = plt.subplots(1, 1)
cax = ax.imshow(m33.z_interp, interpolation='nearest', cmap='coolwarm', origin='lower',
                    extent=[-60,60,-60,60], vmin=-60.0, vmax=60.0)

cbar = fig.colorbar(cax)
cbar.set_label("z location (arcmin)")
plt.savefig("m33_z.pdf", format='pdf')

fig, ax = plt.subplots(1, 1)
cax = ax.imshow(rotdisk.z_interp, interpolation='nearest', cmap='coolwarm', origin='lower',
                    extent=[-60,60,-60,60], vmin=-60.0, vmax=60.0)

cbar = fig.colorbar(cax)
cbar.set_label("z location (arcmin)")
plt.savefig("rotdisk_z.pdf", format='pdf')

#pdb.set_trace()

