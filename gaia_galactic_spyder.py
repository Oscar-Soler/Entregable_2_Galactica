# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:46:35 2023

@author: osole
"""

# astropy imports
import astropy.coordinates as coord
from astropy.table import QTable
import astropy.units as u
from astroquery.gaia import Gaia

# Third-party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# gala imports
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14

import os
#%% Importación de datos del query

os.chdir(str(input('Cambiar de directorio de trabajo a la ubicación del archivo gaia_data.fits: ')))

gaia_data = QTable.read('gaia_data.fits')

dist = coord.Distance(parallax=u.Quantity(gaia_data['parallax']))
dist.min(), dist.max()

c = coord.SkyCoord(ra=gaia_data['ra'],
                   dec=gaia_data['dec'],
                   distance=dist,
                   pm_ra_cosdec=gaia_data['pmra'],
                   pm_dec=gaia_data['pmdec'],
                   radial_velocity=gaia_data['radial_velocity'])

galcen = c.transform_to(coord.Galactocentric(z_sun=0*u.pc,
                                             galcen_distance=8.1*u.kpc))
#%% Coordinate histogram
plt.close(1)
fig, ax = plt.subplots(num = 1)
plt.hist(galcen.y.value, bins = 32) #, bins=np.linspace(-110, 110, 32))
plt.xlabel('$y$ [{0:latex_inline}]'.format(galcen.y.unit));

#%% Coord plot
plt.close(2)
fig, axs = plt.subplots(1, 3, figsize=(21, 6), layout = 'constrained', num = 2)

ax0 = axs[0]
ax = axs[1]
ax1 = axs[2]

ax0.plot(galcen.x.value, galcen.y.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, label = 'Estrellas')
ax0.plot(-galcen.galcen_distance.to(u.pc), 0,
        marker = 'x', linestyle = 'none', label = 'Sol')

#ax.set_xlim(-125, 125)
#ax.set_ylim(200-125, 200+125)

ax0.set_xlabel('$x$ [{0:latex_inline}]'.format(u.pc))
ax0.set_ylabel('$y$ [{0:latex_inline}]'.format(u.pc))
#ax.legend()
ax0.grid()

ax0.set_xticks(np.arange(-8200, -7999, step=50))


ax.plot(galcen.v_x.value, galcen.v_y.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, label = 'Estrellas')
ax.plot(galcen.galcen_v_sun.d_x, galcen.galcen_v_sun.d_y,
        marker = 'x', linestyle = 'none', label = 'Sol')

ax.set_xlim(-300, 300)
ax.set_ylim(200-150, 200+150)

ax.set_xlabel('$v_x$ [{0:latex_inline}]'.format(u.km/u.s))
ax.set_ylabel('$v_y$ [{0:latex_inline}]'.format(u.km/u.s))
#ax.legend()
ax.grid()
#ax.set_title('Velocidad de estrellas en coordenadas galactocéntricas')
#----------------------------------------------------------------------------------
ax1.plot(np.sqrt(galcen.v_x.value**2+galcen.v_y.value**2), galcen.v_z.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, label = 'Estrellas')
ax1.plot(np.sqrt(galcen.galcen_v_sun.d_x.value**2+galcen.galcen_v_sun.d_y.value**2), galcen.galcen_v_sun.d_z.value,
        marker = 'x', linestyle = 'none', label = 'Sol')

ax1.set_xlim(0, 400)
ax1.set_ylim(-150, 150)

ax1.set_xlabel('$\\sqrt{v_x^2 + v_y^2} \\ [\\rm km \\ s^{-1}]$ ')
ax1.set_ylabel('$v_z$ [{0:latex_inline}]'.format(u.km/u.s))
ax1.legend(fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid()
#fig.suptitle('Velocidad de estrellas en coordenadas galactocéntricas')

#%% Diagrama HR, selección de estrellas según masa
M_G = gaia_data['phot_g_mean_mag'] - dist.distmod
BP_RP = gaia_data['phot_bp_mean_mag'] - gaia_data['phot_rp_mean_mag']

plt.close(3)
fig, ax = plt.subplots(1, 1, figsize=(6, 6), num=3)

ax.plot(BP_RP.value, M_G.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.3)

#ax.set_xlim(0, 4)
ax.set_ylim(15, -1)

ax.set_xlabel('$G_{BP}-G_{RP} \\ [\\rm mag]$')
ax.set_ylabel('$M_G \\ [\\rm mag]$')
ax.set_title('Diagrama Color-Magnitud')

np.seterr(invalid="ignore")
hi_mass_mask = ((BP_RP > 0.3*u.mag) & (BP_RP < 0.7*u.mag) &
                (M_G > 2*u.mag) & (M_G < 3.75*u.mag) &
                (np.abs(galcen.v_y - 220*u.km/u.s) < 50*u.km/u.s))

lo_mass_mask = ((BP_RP > 2.55*u.mag) & (BP_RP < 2.8*u.mag) &
                (M_G > 10.3*u.mag) & (M_G < 11*u.mag) &
                (np.abs(galcen.v_y - 220*u.km/u.s) < 50*u.km/u.s))

n_hi_mass = len(galcen[hi_mass_mask])
n_lo_mass = len(galcen[lo_mass_mask])

hi_mass_color = 'tab:blue'
lo_mass_color = 'tab:red'

plt.close(4)
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 100, num = 4)

ax.plot(BP_RP.value, M_G.value,
        color = 'tab:grey', marker='.', markersize = 1, linestyle='none', alpha=0.1, zorder=1)

# Seleccionando region de interés
x_lo_rect = np.array([2.55, 2.8])
y_lo_rect = np.array([10.3,11])

rect_lo = plt.Rectangle((x_lo_rect[0], y_lo_rect[0]), x_lo_rect[1] - x_lo_rect[0], y_lo_rect[1] - y_lo_rect[0], 
                     linewidth=1, edgecolor=lo_mass_color, facecolor='none', zorder=2)
ax.add_patch(rect_lo)

x_hi_rect = np.array([0.3,0.7])
y_hi_rect = np.array([2,3.75])

rect_hi = plt.Rectangle((x_hi_rect[0], y_hi_rect[0]), x_hi_rect[1] - x_hi_rect[0], y_hi_rect[1] - y_hi_rect[0], 
                     linewidth=1, edgecolor=hi_mass_color, facecolor='none', zorder=2)
ax.add_patch(rect_hi)


for mask, color, lab, n in zip([lo_mass_mask, hi_mass_mask],
                       [lo_mass_color, hi_mass_color], ['Baja masa (%i)', 'Alta masa (%i)'], [n_lo_mass, n_hi_mass]):
    ax.plot(BP_RP[mask].value, M_G[mask].value,
            marker='.', markersize = 1, linestyle='none',
            alpha=0.5, color=color, label = lab%(n))

#ax.set_xlim(0, 3)
ax.set_ylim(15, -1)

ax.set_xlabel('$G_{\\rm BP}-G_{\\rm RP} \\ [\\rm mag]$')
ax.set_ylabel('$M_{\\rm G} \\ [\\rm mag]$')
ax.set_title('Diagrama Color-Magnitud')
legend = plt.legend(markerscale = 10)

#%% Coords plot masses
plt.close(5)
fig, axs = plt.subplots(1, 3, figsize=(21, 6), layout = 'constrained', num=5)

ax0 = axs[0]
ax = axs[1]
ax1 = axs[2]

ax0.plot(galcen[lo_mass_mask].x.value, galcen[lo_mass_mask].y.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, color = lo_mass_color, label = 'Baja masa')
ax0.plot(galcen[hi_mass_mask].x.value, galcen[hi_mass_mask].y.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, color = hi_mass_color, label = 'Alta masa')
ax0.plot(-galcen.galcen_distance.to(u.pc), 0,
        marker = 'x', linestyle = 'none', label = 'Sol', color = 'orange')

#ax.set_xlim(-125, 125)
#ax.set_ylim(200-125, 200+125)

ax0.set_xlabel('$x$ [{0:latex_inline}]'.format(u.pc))
ax0.set_ylabel('$y$ [{0:latex_inline}]'.format(u.pc))
#ax.legend()
ax0.grid()

ax0.set_xticks(np.arange(-8200, -7999, step=50))

#-----------------------------------------

ax.plot(galcen[lo_mass_mask].v_x.value, galcen[lo_mass_mask].v_y.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, color = lo_mass_color, label = 'Baja masa')
ax.plot(galcen[hi_mass_mask].v_x.value, galcen[hi_mass_mask].v_y.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, color = hi_mass_color, label = 'Alta masa')
ax.plot(galcen.galcen_v_sun.d_x, galcen.galcen_v_sun.d_y,
        marker = 'x', linestyle = 'none', label = 'Sol', color = 'orange')

ax.set_xlim(-150, 150)
ax.set_ylim(150, 300)

ax.set_xlabel('$v_x$ [{0:latex_inline}]'.format(u.km/u.s))
ax.set_ylabel('$v_y$ [{0:latex_inline}]'.format(u.km/u.s))
#ax.legend()
ax.grid()
#ax.set_title('Velocidad de estrellas en coordenadas galactocéntricas')
#----------------------------------------------------------------------------------
ax1.plot(np.sqrt(galcen[lo_mass_mask].v_x.value**2+galcen[lo_mass_mask].v_y.value**2), galcen[lo_mass_mask].v_z.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, color = lo_mass_color, label = 'Baja masa')
ax1.plot(np.sqrt(galcen[hi_mass_mask].v_x.value**2+galcen[hi_mass_mask].v_y.value**2), galcen[hi_mass_mask].v_z.value,
        marker='.', markersize = 1, linestyle='none', alpha=0.5, color = hi_mass_color, label = 'Alta masa')
ax1.plot(np.sqrt(galcen.galcen_v_sun.d_x.value**2+galcen.galcen_v_sun.d_y.value**2), galcen.galcen_v_sun.d_z.value,
        marker = 'x', linestyle = 'none', label = 'Sol', color = 'orange')

ax1.set_xlim(150, 300)
ax1.set_ylim(-100, 100)

ax1.set_xlabel('$\\sqrt{v_x^2 + v_y^2} \\ [\\rm km \\ s^{-1}]$ ')
ax1.set_ylabel('$v_z$ [{0:latex_inline}]'.format(u.km/u.s))
ax1.legend(fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid()
#fig.suptitle('Velocidad de estrellas en coordenadas galactocéntricas')

#%% Potentials
milky_way = gp.MilkyWayPotential()
different_disk_potential = gp.MilkyWayPotential(disk=dict(m=8e11*u.Msun, a = 3*u.kpc, b=0.28*u.kpc))


def plot_potential(pot, x_len = 100, y_len = 100, xrange = [-10,10], yrange = [-10,10],
                   fnum = 6, pot_name = 'Potencial de la Vía Láctea', contour = False, cv = False):

    x = np.linspace(xrange[0], xrange[1], x_len)
    y = np.linspace(yrange[0], yrange[1], y_len)
    z = np.zeros([x_len,y_len])

    X, Y = np.meshgrid(x,y)
    Z = pot([X,Y,z])
    
    plt.close(fnum)
    fig = plt.figure(num = fnum)
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z.value, 50, cmap='binary')
    ax.plot_surface(X, Y, Z.value, rstride = 1, cstride = 1, cmap='viridis')
    ax.set_xlabel('$x$ [kpc]')
    ax.set_ylabel('$y$ [kpc]')
    ax.set_zlabel('$[\\rm{kpc}^2/\\rm{Myr}^2]$')
    ax.set_title(pot_name)
    fig.tight_layout()
    
    if cv == True:
        plt.close(fnum+1)
        fig = plt.figure(num = fnum+1)
        c_v2 = pot.circular_velocity([x,z[0],z[0]])
        plt.plot(x, c_v2)
        fig.axes[0].set_xlim(0,max(x))
        plt.xlabel('$\\rho$ [kpc]')
        plt.ylabel('$v$ [km/s]')

plot_potential(milky_way, fnum = 6, cv=True)
plot_potential(different_disk_potential, fnum = 8, pot_name = 'Potencial de la Vía Láctea con el disco modificado', cv = True)

# different_nuc_potential = gp.MilkyWayPotential(nucleus=dict(m=1e10*u.Msun, c = 1*u.kpc))
# plot_potential(different_nuc_potential, fnum = 30, pot_name = 'Potencial de la Vía Láctea con el núcleo modificado', cv = True)

#%%
different_nuc_potential_2 = gp.MilkyWayPotential(nucleus=dict(m=1e10*u.Msun, c = 0.05*u.kpc))
plot_potential(different_nuc_potential_2, fnum = 61, pot_name = 'Potencial de la Vía Láctea con el núcleo modificado', cv = True)
#%% Orbits
def orbit_integration(potential, galcen_ini = galcen, hi_m_mask = hi_mass_mask, lo_m_mask = lo_mass_mask):
    H = gp.Hamiltonian(potential)
    w0_hi = gd.PhaseSpacePosition(galcen_ini[hi_m_mask].cartesian)
    w0_lo = gd.PhaseSpacePosition(galcen_ini[lo_m_mask].cartesian)
    orbs_hi = H.integrate_orbit(w0_hi, dt=1*u.Myr,
                                  t1=0*u.Myr, t2=500*u.Myr)
    orbs_lo = H.integrate_orbit(w0_lo, dt=1*u.Myr,
                                  t1=0*u.Myr, t2=500*u.Myr)
    return orbs_hi, orbs_lo

orbits_hi, orbits_lo = orbit_integration(milky_way)
orbits_hi_2, orbits_lo_2 = orbit_integration(different_disk_potential)
# orbits_hi_3, orbits_lo_3 = orbit_integration(different_nuc_potential)
orbits_hi_4, orbits_lo_4 = orbit_integration(different_nuc_potential_2)

#%% zmax, peri, apo, ecc
def calculations(orb_hi, orb_lo):
    zm_hi = orb_hi.zmax(approximate=True)
    print('High-mass zmax calculated')
    zm_lo = orb_lo.zmax(approximate=True)
    print('Low-mass zmax calculated')
    
    e_hi = orb_hi.eccentricity(approximate=True)
    print('High-mass eccentricity calculated')
    e_lo = orb_lo.eccentricity(approximate=True)
    print('Low-mass eccentricity calculated')
    
    return zm_hi,zm_lo, e_hi, e_lo#a_hi, a_lo, p_hi, p_lo, e_hi, e_lo

zmax_hi, zmax_lo, ecc_hi, ecc_lo = calculations(orbits_hi, orbits_lo)
zmax_hi_2, zmax_lo_2, ecc_hi_2, ecc_lo_2 = calculations(orbits_hi_2, orbits_lo_2)
# zmax_hi_3, zmax_lo_3, ecc_hi_3, ecc_lo_3 = calculations(orbits_hi_3, orbits_lo_3)
zmax_hi_4, zmax_lo_4, ecc_hi_4, ecc_lo_4 = calculations(orbits_hi_4, orbits_lo_4)

#%% zmax, ecc selection
def indexes(zm_hi, zm_lo, ec_hi, ec_lo, orbs_hi, orbs_lo, ploteo = False, ploteo_3d = False):
    zm_hi = zm_hi.value
    zm_lo = zm_lo.value
    avzm_hi, avzm_lo = np.mean(zm_hi), np.mean(zm_lo)
    argmaxzm_hi, argmaxzm_lo = np.argmax(zm_hi), np.argmax(zm_lo)
    argminzm_hi, argminzm_lo = np.argmin(zm_hi), np.argmin(zm_lo)
    argavzm_hi, argavzm_lo = np.argmin(abs(zm_hi-avzm_hi)), np.argmin(abs(zm_lo-avzm_lo))
    
    avec_hi, avec_lo = np.mean(ec_hi), np.mean(ec_lo)
    argmaxec_hi, argmaxec_lo = np.argmax(ec_hi), np.argmax(ec_lo)
    argminec_hi, argminec_lo = np.argmin(ec_hi), np.argmin(ec_lo)
    argavec_hi, argavec_lo = np.argmin(abs(ec_hi-avec_hi)), np.argmin(abs(ec_lo-avec_lo))
    
    argav_hi, argav_lo = np.argmin(abs((ec_hi-avec_hi)**2+(zm_hi-avzm_hi)**2)), np.argmin(abs((ec_lo-avec_lo)**2+(zm_lo-avzm_lo)**2))
    
    if ploteo == True:
        fig = orbs_hi[:, argmaxzm_hi].plot(color=hi_mass_color, label = 'Alta masa')
        _ = orbs_lo[:, argmaxzm_lo].plot(axes=fig.axes, color=lo_mass_color, label = 'Baja masa')
        fig.suptitle('Órbitas con $z_{\\rm max}$ máximo \n Alta masa: $z_{\\rm max}=%.2f$ [kpc],  Baja masa: $z_{\\rm max}=%.2f$ [kpc]'%(zm_hi[argmaxzm_hi], zm_lo[argmaxzm_lo]))
        fig.legend(['Alta masa', 'Baja masa'])
        
        fig = orbs_hi[:, argmaxec_hi].plot(color=hi_mass_color)
        _ = orbs_lo[:, argmaxec_lo].plot(axes=fig.axes, color=lo_mass_color)
        fig.suptitle('Órbitas con excentricidad máxima \n Alta masa: $\\epsilon=%.2f$,  Baja masa: $\\epsilon=%.2f$'%(ec_hi[argmaxec_hi], ec_lo[argmaxec_lo]))
        fig.legend(['Alta masa', 'Baja masa'])

        fig = orbs_hi[:, argminzm_hi].plot(color=hi_mass_color)
        _ = orbs_lo[:, argminzm_lo].plot(axes=fig.axes, color=lo_mass_color)
        fig.suptitle('Órbitas con $z_{\\rm max}$ mínimo \n Alta masa: $z_{\\rm max}=%.2f$ [kpc],  Baja masa: $z_{\\rm max}=%.2f$ [kpc]'%(zm_hi[argminzm_hi], zm_lo[argminzm_lo]))
        fig.legend(['Alta masa', 'Baja masa'])

        fig = orbs_hi[:, argminec_hi].plot(color=hi_mass_color)
        _ = orbs_lo[:, argminec_lo].plot(axes=fig.axes, color=lo_mass_color)
        fig.suptitle('Órbitas con excentricidad mínima \n Alta masa: $\\epsilon=%.2f$,  Baja masa: $\\epsilon=%.2f$'%(ec_hi[argminec_hi], ec_lo[argminec_lo]))
        fig.legend(['Alta masa', 'Baja masa'])

        fig = orbs_hi[:, argavzm_hi].plot(color=hi_mass_color)
        _ = orbs_lo[:, argavzm_lo].plot(axes=fig.axes, color=lo_mass_color)
        fig.suptitle('Órbitas con $z_{\\rm max}$ promedio \n Alta masa: $z_{\\rm max}=%.2f$ [kpc],  Baja masa: $z_{\\rm max}=%.2f$ [kpc]'%(zm_hi[argavzm_hi], zm_lo[argavzm_lo]))
        fig.legend(['Alta masa', 'Baja masa'])

        fig = orbs_hi[:, argavec_hi].plot(color=hi_mass_color)
        _ = orbs_lo[:, argavec_lo].plot(axes=fig.axes, color=lo_mass_color)
        fig.suptitle('Órbitas con excentricidad promedio \n Alta masa: $\\epsilon=%.2f$,  Baja masa: $\\epsilon=%.2f$'%(ec_hi[argavec_hi], ec_lo[argavec_lo]))
        fig.legend(['Alta masa', 'Baja masa'])

        fig = orbs_hi[:, argav_hi].plot(color=hi_mass_color)
        _ = orbs_lo[:, argav_lo].plot(axes=fig.axes, color=lo_mass_color)
        fig.suptitle('Órbitas con $z_{\\rm max}$ y $\\epsilon$ promedio \n Alta masa: $z_{\\rm max}=%.2f$ [kpc], $\\epsilon=%.2f$;  Baja masa: $z_{\\rm max}=%.2f$ [kpc], $\\epsilon=%.2f$'%(zm_hi[argav_hi], ec_hi[argav_hi], zm_lo[argav_lo], ec_lo[argav_lo]))
        fig.legend(['Alta masa', 'Baja masa'])
        
    if ploteo_3d == True:
        fig = orbs_hi[:, argmaxzm_hi].plot_3d(color=hi_mass_color, label = 'Alta masa')
        _ = orbs_lo[:, argmaxzm_lo].plot_3d(ax=fig[1], color=lo_mass_color, label = 'Baja masa')
        fig[1].set_title('Órbitas con $z_{\\rm max}$ máximo \n Alta masa: $z_{\\rm max}=%.2f$ [kpc] \n Baja masa: $z_{\\rm max}=%.2f$ [kpc]'%(zm_hi[argmaxzm_hi], zm_lo[argmaxzm_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)
        
        fig = orbs_hi[:, argmaxec_hi].plot_3d(color=hi_mass_color)
        _ = orbs_lo[:, argmaxec_lo].plot_3d(ax=fig[1], color=lo_mass_color)
        fig[1].set_title('Órbitas con excentricidad máxima \n Alta masa: $\\epsilon=%.2f$ \n Baja masa: $\\epsilon=%.2f$'%(ec_hi[argmaxec_hi], ec_lo[argmaxec_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)

        fig = orbs_hi[:, argminzm_hi].plot_3d(color=hi_mass_color)
        _ = orbs_lo[:, argminzm_lo].plot_3d(ax=fig[1], color=lo_mass_color)
        fig[1].set_title('Órbitas con $z_{\\rm max}$ mínimo \n Alta masa: $z_{\\rm max}=%.2f$ [kpc] \n Baja masa: $z_{\\rm max}=%.2f$ [kpc]'%(zm_hi[argminzm_hi], zm_lo[argminzm_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)

        fig = orbs_hi[:, argminec_hi].plot_3d(color=hi_mass_color)
        _ = orbs_lo[:, argminec_lo].plot_3d(ax=fig[1], color=lo_mass_color)
        fig[1].set_title('Órbitas con excentricidad mínima \n Alta masa: $\\epsilon=%.2f$ \n Baja masa: $\\epsilon=%.2f$'%(ec_hi[argminec_hi], ec_lo[argminec_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)

        fig = orbs_hi[:, argavzm_hi].plot_3d(color=hi_mass_color)
        _ = orbs_lo[:, argavzm_lo].plot_3d(ax=fig[1], color=lo_mass_color)
        fig[1].set_title('Órbitas con $z_{\\rm max}$ promedio \n Alta masa: $z_{\\rm max}=%.2f$ [kpc] \n Baja masa: $z_{\\rm max}=%.2f$ [kpc]'%(zm_hi[argavzm_hi], zm_lo[argavzm_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)

        fig = orbs_hi[:, argavec_hi].plot_3d(color=hi_mass_color)
        _ = orbs_lo[:, argavec_lo].plot_3d(ax=fig[1], color=lo_mass_color)
        fig[1].set_title('Órbitas con excentricidad promedio \n Alta masa: $\\epsilon=%.2f$ \n Baja masa: $\\epsilon=%.2f$'%(ec_hi[argavec_hi], ec_lo[argavec_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)

        fig = orbs_hi[:, argav_hi].plot_3d(color=hi_mass_color)
        _ = orbs_lo[:, argav_lo].plot_3d(ax=fig[1], color=lo_mass_color)
        fig[1].set_title('Órbitas con $z_{\\rm max}$ y $\\epsilon$ promedio \n Alta masa: $z_{\\rm max}=%.2f$ [kpc], $\\epsilon=%.2f$ \n Baja masa: $z_{\\rm max}=%.2f$ [kpc], $\\epsilon=%.2f$'%(zm_hi[argav_hi], ec_hi[argav_hi], zm_lo[argav_lo], ec_lo[argav_lo]))
        fig[1].legend(['Alta masa', 'Baja masa'])
        fig[1].set_zlim(-5,5)

        
    
# indexes(zmax_hi, zmax_lo, ecc_hi, ecc_lo, orbits_hi, orbits_lo, ploteo = True)  
indexes(zmax_hi, zmax_lo, ecc_hi, ecc_lo, orbits_hi, orbits_lo, ploteo = True, ploteo_3d = True)  

#%%
indexes(zmax_hi_2, zmax_lo_2, ecc_hi_2, ecc_lo_2, orbits_hi_2, orbits_lo_2, ploteo = True, ploteo_3d = False)  
# indexes(zmax_hi_3, zmax_lo_3, ecc_hi_3, ecc_lo_3, orbits_hi_3, orbits_lo_3, ploteo = True, ploteo_3d = False)  
indexes(zmax_hi_4, zmax_lo_4, ecc_hi_4, ecc_lo_4, orbits_hi_4, orbits_lo_4, ploteo = True, ploteo_3d = False)  

#%% histograms
def plot_histogram(data_lo, data_hi, x_lab, xmin=0, xmax=2, nbins = 50, n_lo = n_lo_mass, n_hi = n_hi_mass,
                   fsz = (7,5), fnum = 1, title = 'Title'):
    plt.close(fnum)
    bins = np.linspace(xmin, xmax, nbins)
    plt.figure(figsize=fsz, num = fnum)
    plt.hist(data_lo.value, bins=bins,
             alpha=0.4, density=False, label='Baja masa (%i)'%n_lo,
             color=lo_mass_color)
    plt.hist(data_hi.value, bins=bins,
             alpha=0.4, density=False, label='Alta masa (%i)'%n_hi,
             color=hi_mass_color)
    plt.legend(loc='best', fontsize=14)
    # plt.yscale('log')
    plt.xlabel(x_lab)
    plt.title(title)

fnum_cell = 25
for i in range(5):
    plt.close(fnum_cell+i)
title1 = 'Potencial de la Vía Láctea'
title2 = 'Potencial de la Vía Láctea con el disco modificado'
# title3 = 'Potencial de la Vía Láctea con el núcleo modificado'
title4 = 'Potencial de la Vía Láctea con el núcleo modificado'
plot_histogram(zmax_lo, zmax_hi, '$z_{\\rm max}$ [kpc]', fnum = fnum_cell, title = title1)
plot_histogram(ecc_lo, ecc_hi, 'Excentricidad $\\epsilon$', xmax = 0.5, fnum = fnum_cell+1, title = title1)
plot_histogram(zmax_lo_2, zmax_hi_2, '$z_{\\rm max}$ [kpc]', xmax = 0.5, fnum = fnum_cell+2, title = title2)
plot_histogram(ecc_lo_2, ecc_hi_2, 'Excentricidad $\\epsilon$', xmin = min(min(ecc_lo_2), min(ecc_hi_2)), xmax = max(max(ecc_lo_2), max(ecc_hi_2)), fnum = fnum_cell+3, title = title2)
# plot_histogram(zmax_lo_3, zmax_hi_3, '$z_{\\rm max}$ [kpc]', xmax = 5, fnum = 51, title = title3)
# plot_histogram(ecc_lo_3, ecc_hi_3, 'Excentricidad $\\epsilon$', xmin = min(min(ecc_lo_3), min(ecc_hi_3)), xmax = max(max(ecc_lo_3), max(ecc_hi_3)), fnum = 52, title = title3)
plot_histogram(zmax_lo_4, zmax_hi_4, '$z_{\\rm max}$ [kpc]', fnum = 71, title = title4)
plot_histogram(ecc_lo_4, ecc_hi_4, 'Excentricidad $\\epsilon$', xmin = min(min(ecc_lo_4), min(ecc_hi_4)), xmax = max(max(ecc_lo_4), max(ecc_hi_4)), fnum = 72, title = title4)

#------------
#%%

def plot_zmax_ecc(zm_hi, zm_lo, e_hi, e_lo, fignum = 1, title = '',
                  xmax=0.4, xmin=0, ymin = 0, ymax = 1.5):
    plt.close(fignum+6)
    fig, ax = plt.subplots(layout= 'tight', figsize = (10,6), num = fignum+6)
    
    ax.plot(e_lo, zm_lo, 
             marker = '.', ms = 3, linestyle = 'none', alpha = .5, color = lo_mass_color,
             label = 'Baja masa (%i)'%n_lo_mass)
    
    ax.plot(e_hi, zm_hi, 
             marker = '.', ms = 3, linestyle = 'none', alpha = .5, color = hi_mass_color,
             label = 'Alta masa (%i)'%n_hi_mass)
    
    ax.axvline(np.mean(e_hi), color = 'white', linewidth = 4, alpha = 0.8)
    ax.axvline(np.mean(e_hi), linestyle ='--', color = hi_mass_color)
    ax.axvline(np.mean(e_lo), color = 'white', linewidth = 4, alpha = 0.8)
    ax.axvline(np.mean(e_lo), linestyle ='--', color = lo_mass_color)
    ax.axhline(np.mean(zm_hi.value), color = 'white', linewidth = 4, alpha = 0.8)
    ax.axhline(np.mean(zm_hi.value), linestyle ='--', color = hi_mass_color)
    ax.axhline(np.mean(zm_lo.value), color = 'white', linewidth = 4, alpha = 0.8)
    ax.axhline(np.mean(zm_lo.value), linestyle ='--', color = lo_mass_color)
    ax.plot(np.mean(e_hi), np.mean(zm_hi.value), 'k+')
    ax.plot(np.mean(e_lo), np.mean(zm_lo.value), 'k+')
    
    xpos = (xmax-xmin)*2/3+xmin
    ypos = (ymax-ymin)/2+ymin
    # print(xlen, ylen)
    xlen = (xmax-xmin)/4*1.2
    ylen = (ymax-ymin)/4

    box = matplotlib.patches.Rectangle((xpos, ypos), xlen, ylen, linewidth=1, edgecolor='black', facecolor='white', zorder=2)
    ax.add_patch(box)
    
    # Add text inside the box
    
    text = '$\\langle z_{\\rm max} \\rangle = %.2f$ kpc, $\\langle \\epsilon \\rangle = %.2f$'%(np.mean(zm_lo.value), np.mean(e_lo))
    ax.text(xpos+xlen/2, ypos+ylen/2+ylen*.2, text, ha='center', va='center', fontsize=12, color=lo_mass_color)
    
    text = '$\\langle z_{\\rm max} \\rangle = %.2f$ kpc, $\\langle \\epsilon \\rangle = %.2f$'%(np.mean(zm_hi.value), np.mean(e_hi))
    ax.text(xpos+xlen/2, ypos+ylen/2-ylen*.2, text, ha='center', va='center', fontsize=12, color=hi_mass_color)
    
    
    ax.set_xlabel('Excentrididad $\\epsilon$')
    ax.set_ylabel('$z_{\\rm max}$ [kpc]')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    # plt.legend(markerscale = 10)
    ax.legend(loc = 'upper right', markerscale = 10)
    # plt.title(title)

plot_zmax_ecc(zmax_hi, zmax_lo, ecc_hi, ecc_lo, fignum = fnum_cell+6, title=title1)
plot_zmax_ecc(zmax_hi_2, zmax_lo_2, ecc_hi_2, ecc_lo_2, fignum = 53, title=title2, ymax = 0.5, xmin = float(min(ecc_hi_2)), xmax = float(max(ecc_lo_2)))
# plot_zmax_ecc(zmax_hi_3, zmax_lo_3, ecc_hi_3, ecc_lo_3, fignum = 54, title=title3)
plot_zmax_ecc(zmax_hi_4, zmax_lo_4, ecc_hi_4, ecc_lo_4, fignum = 74, title=title4)

#%% mean, std
mean_zs = np.array([np.mean(zmax_hi.value), np.mean(zmax_hi_2.value), np.mean(zmax_hi_4.value)])
mean_eccs = np.array([np.mean(ecc_hi.value), np.mean(ecc_hi_2.value), np.mean(ecc_hi_4.value)])
std_zs = np.array([np.std(zmax_hi.value), np.std(zmax_hi_2.value), np.std(zmax_hi_4.value)])
std_eccs = np.array([np.std(ecc_hi.value), np.std(ecc_hi_2.value), np.std(ecc_hi_4.value)])

for name, arr_m, arr_std in zip(['z', 'ecc'],[mean_zs, mean_eccs],[std_zs, std_eccs]):
    print(name+', %.2f  %.2f, %.2f %.2f, %.2f %.2f'%(arr_m[0], arr_std[0], arr_m[1], arr_std[1], arr_m[2], arr_std[2]))
    
print()
print()

mean_zs = np.array([np.mean(zmax_lo.value), np.mean(zmax_lo_2.value), np.mean(zmax_lo_4.value)])
mean_eccs = np.array([np.mean(ecc_lo.value), np.mean(ecc_lo_2.value), np.mean(ecc_lo_4.value)])
std_zs = np.array([np.std(zmax_lo.value), np.std(zmax_lo_2.value), np.std(zmax_lo_4.value)])
std_eccs = np.array([np.std(ecc_lo.value), np.std(ecc_lo_2.value), np.std(ecc_lo_4.value)])

for name, arr_m, arr_std in zip(['z', 'ecc'],[mean_zs, mean_eccs],[std_zs, std_eccs]):
    print(name+', %.2f  %.2f, %.2f %.2f, %.2f %.2f'%(arr_m[0], arr_std[0], arr_m[1], arr_std[1], arr_m[2], arr_std[2]))
    
