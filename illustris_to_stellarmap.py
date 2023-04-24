import os
import sys
import h5py 
import requests
import json as j
import numpy as np
import pandas as pd

from orientation import *
from utils import *

from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
from sklearn.neighbors import KDTree

h = 0.6774
cosmo = LambdaCDM(H0=100*h, Om0=0.3089, Ob0=0.0486, Ode0=0.6911, Tcmb0=2.73)

snapZ = np.load('snapTNG.npy')
headers = {"api-key":"ff352a2affacf64753689dd603b5b44e"} #replace with your API-KEY

def solve_coordinates(X, cm, rotation, simulation='TNG50-1'):
    rotated = rotation.dot((correct_periodic_distances(X, simulation) - correct_periodic_distances(cm, simulation)).T).T * u.kpc
    coords = rotated.to(u.pc) * a / cosmo.h
    x = np.squeeze(np.array(coords.T[0], float))
    y = np.squeeze(np.array(coords.T[1], float))
    z = np.squeeze(np.array(coords.T[2], float))
    return x, y, z

def solve_velocities(V, simulation='TNG50-1'):
    V *= np.sqrt(a)
    vx = V.T[0].astype(int)
    vy = V.T[1].astype(int)
    vz = V.T[2].astype(int)
    return vx, vy, vz

def load_stellar(stellar, shPos, snap):

    smoothing_rank = 64

    age_a = stellar['GFM_StellarFormationTime'][:]
    stars = np.where(age_a > 0)[0] # exclude stellar wind entries
    ages = cosmo.lookback_time(1/stellar['GFM_StellarFormationTime'][:][stars] - 1) - cosmo.lookback_time(snapZ.T[snap][1])
    print('Loading stellar particles')
    
    x, y, z = solve_coordinates(stellar['Coordinates'][:][stars], shPos, rotation, simulation=simulation)


    within_fov = np.where((abs(x) < FOV*1000/2) & (abs(y) <  FOV*1000/2) & (abs(z) <  FOV*1000/2))
    x = x[within_fov]
    y = y[within_fov]
    z = z[within_fov]

    age_a = age_a[within_fov]
    ages = ages[within_fov]

    print(f'Found {within_fov[0].shape} stars in FOV')
    #vx, vy, vz = solve_velocitin
    h, ind = smoothing_length(x, y, z, Nk=smoothing_rank, return_index=True)

    mass = 10**10 * u.solMass * stellar['GFM_InitialMass'][:][stars][within_fov] / cosmo.h
    metalicity = stellar['GFM_Metallicity'][:][stars][within_fov]

    
    '''
        Stellar particles in Illustris don't have a density associated to it
        (or parent gas particle), so we use the density estimated by the 
        smoothing length
        https://www.tng-project.org/data/forum/topic/251/how-to-get-the-mass-stellar-volume-density-data/
        https://arxiv.org/pdf/1203.5667.pdf
    '''
    print('Calculating density')
    volume = 4/3 * np.pi * (h*u.pc)**3 
    rho = smoothing_rank * mass[ind].mean(axis=1) / volume
    rho = rho.to(u.solMass / u.pc**3)

    try:
        H_abundance = gas['GFM_Metals'][:][stars][within_fov][:, 0]
    except:
        H_abundance = np.zeros_like(mass.value) + 0.75


    nH = (H_abundance * rho.to(u.kg/u.cm**3) / const.m_p) # neutral hydrogen number density
    
    print("polytropic pressure")
    #normalization for polytropic pressure
    gamma = 4/3
    norm = 10**3 * u.cm**(-3) * u.K * 1/((0.1 * u.cm**-3 * const.m_p / 0.75)**(gamma) / const.k_B)
    P = (norm * rho**(gamma)).to(u.Pa)

    print("SFR")
    # SFR derived from the pressure Schaye et al. (2015)
    A = 1.515e-4 * u.solMass * u.yr**(-1) * u.kpc**(-2)
    n_p = 1.4
    SFR = mass * A * (u.solMass * u.pc**(-2))**(-n_p) * ((5/3)/const.G * P)**((n_p-1)/2)
    SFR = SFR.to(u.solMass / u.yr)
    
    starburst_threhold = (100 * u.Myr).to(u.yr)
    #SFR = mass.value / starburst_threhold.to(u.yr).value # assume constant SFR
    logC = 3/5 * np.log10((mass).to(u.kg) / const.M_sun) + 2/5 * np.log10(P/const.k_B / (u.cm**-3 * u.K))
    fc = np.zeros_like(logC) + 0.2
    T = np.zeros_like(fc)
    # this is the table with all stellar particles without distinction. We separate them in GALEXEV and MAPPINGS
    # when the resampling process takes place

    
    

    df_all = pd.DataFrame(np.array([x, y, z, h, mass, metalicity, ages.to(u.yr).value, H_abundance, rho, nH, P, logC, SFR, fc, T]).T,
                        columns=['x', 'y', 'z', 'h', 'mass', 'Z', 'age', 'Hab', 'density', 'nH', 'P', 'logC', 'SFR', 'fc', 'T'], dtype=object)
                

    df_all['type'] = 'stellar'

    return df_all      

def load_particles(stellar, gas, shPos, snap,  simulation='TNG50-1'):
    
    """
        With All Combined
    """
    df_all = load_stellar(stellar, shPos, snap)   
   
    within_fov = np.where((abs(df_all.x) < FOV*1000/2) & (abs(df_all.y) <  FOV*1000/2) & (abs(df_all.z) <  FOV*1000/2))

    df_all = df_all.iloc[within_fov]
    
    starburst_threshold = (10 * u.Myr).to(u.yr).value
    regular = np.where(df_all.age > starburst_threshold) #GALAXEV
    
    galaxev_df = []
    galaxev_df = df_all[['x', 'y', 'z', 'h', 'mass', 'Z', 'age']].iloc[regular]
    
    return galaxev_df

def sample_about(x, y, z, r):

    phi = np.random.uniform(0, 2*np.pi, len(x))
    costheta = np.random.uniform(-1,1, len(x))
    u = np.random.uniform(0,1, len(x))

    theta = np.arccos(costheta)
    ri = r * np.cbrt(u)
    xn = x + ri * np.sin(theta) * np.cos(phi)
    yn = y + ri * np.sin(theta) * np.sin(phi)
    zn = z + ri * np.cos(theta)

    return xn, yn, zn

def sample_splined(x, y, z, r):

    xn = x + r * np.random.normal(scale=0.29, size=len(r))
    yn = y + r * np.random.normal(scale=0.29, size=len(r))
    zn = z + r * np.random.normal(scale=0.29, size=len(r))

    return xn, yn, zn

def stochResamp(sfri, mi):
    
    # mass resampling parameters (see Kennicutt & Evans 2012 section 2.5)
    m_min = 700         # minimum mass of sub-particle in M_solar
    m_max = 1e6         # maximum mass of sub-particle in M_solar
    alpha = 1.8         # exponent of power-law mass function
    alpha1 = 1. - alpha

    # age resampling parameters
    thresh_age = 1e8    # period over which to resample in yr (100 Myr)

    # initialise lists for output
    ms   = [[]]
    ts   = [[]]
    idxs = [[]]
    mdiffs = []

    # for each parent particle, determine the star-forming sub-particles

    # determine the maximum number of sub-particles based on the minimum sub-particle mass
    N = int(max(1,np.ceil(mi/m_min)))

    # generate random sub-particle masses from a power-law distribution between min and max values
    X = np.random.random(N)
    m = (m_min**alpha1 + X*(m_max**alpha1-m_min**alpha1))**(1./alpha1)

    # limit and normalize the list of sub-particles to the total mass of the parent
    mlim = m[np.cumsum(m)<=mi]
    if len(mlim)<1: mlim = m[:1]
    m = mi/mlim.sum() * mlim
    N = len(m)

    # generate random decay lookback time for each sub-particle
    X = np.random.random(N)               # X in range (0,1]
    t = thresh_age + mi/sfri * np.log(1-X)

    # determine mask for sub-particles that form stars by present day
    issf = t > 0.

    # add star-forming sub-particles to the output lists
    
    return m, t/1e6

def resampler_generator(sf_particles):
    
    resampling_threshold = 1e8 * u.yr
    for x, y, z, h, mass, Z, age, Hab, density, nH, P, logC, SFR, fc, T, typel in sf_particles:

        subparticle_masses, formation_time = stochResamp(SFR, mass)
        #subparticle_masses = imf.make_cluster(mass, silent=True, massfunc='heyer', stop_criterion='before', mmin=700, mmax=1e6)

        #if len(subparticle_masses) > 0:
        #    subparticle_masses = subparticle_masses[:-1]
        #    subparticle_masses /= subparticle_masses.sum()
        #    subparticle_masses *= mass
        #else:
        #    continue

        #n_resampled = len(subparticle_masses)
        #X = np.random.random(n_resampled)  
        #formation_time = resampling_threshold  + mass/SFR * np.log(1-X) * u.yr  
        #formation_time = (formation_time).to(u.Myr).value
        try:
            yield x, y, z, h, mass, Z, age, Hab, density, nH, P, SFR, fc, T, subparticle_masses, formation_time   
        except StopIteration:
            return

def resample_numpy(all_df):
    
    resampling_threshold = 1e8 * u.yr # 100 Myrs

    old_stars_idx = np.where(all_df.age >= resampling_threshold.value)
    old_stars = all_df[['x', 'y', 'z', 'h', 'mass', 'Z', 'age']].iloc[old_stars_idx]
    galaxev_dfs  = [np.array([old_stars.x, old_stars.y, old_stars.z, old_stars.h, old_stars.mass, old_stars.Z, old_stars.age])]
    
    #non-starforming gass, these particles are not resampled
    ISM = np.where((all_df.SFR == 0) & (all_df.type == 'gas'))
    dust_ism = all_df.iloc[ISM]
    dust_ism_dfs = [np.array([dust_ism.x, dust_ism.y, dust_ism.z, dust_ism.h, dust_ism.mass, dust_ism.Z,   dust_ism.SFR,  dust_ism['T']])]#all_df[['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass', 'T']].iloc[ISM]]

    # this will be populated after the resampling process
    mappings_dfs = []
    
    '''
        particles that contribute to SF
    '''
    SF_particles_idx = np.where((all_df.age < resampling_threshold.value) & 
                               (all_df.SFR > 0))
    
    sf_particles = all_df.iloc[SF_particles_idx].to_numpy()
    
    # how much mass on average these particles form over 100 Myr
    # this is used in the stochastic process to assign formation
    # times
    #mass_to_form = sf_particles.SFR * (100 * u.Myr).to(u.yr).value
    """
         0    1    2    3     4    5     6      7      8     9      10       11      12    13    14      15    16    17
       ['x', 'y', 'z', 'h', 'mass', 'Z', 'age', 'Hab', 'density', 'nH', 'P', 'logC', 'SFR', 'fc', 'T']
       ['x', 'y', 'z', 'h', 'mass', 'Z', 'age', 'Hab', 'density', 'nH', 'P', 'logC', 'SFR', 'fc', 'T']
    """
    nstars = sf_particles.shape[0]
    formed_mass = np.zeros(nstars)
    print(f'Resampling {nstars} stars')
    for i, (x, y, z, h, mass, Z, age, Hab, density, nH, P,  SFR, fc, T, subparticle_masses, formation_time) in enumerate(resampler_generator(sf_particles)):

        n_resampled = len(subparticle_masses)

        if i % 5000 == 0:
            print(f'{i}/{nstars}')

        to_dusty_ism = np.where((formation_time < 0))
        mappings_idx = np.where((formation_time > 0) & (formation_time <= 10))
        galexev_idx = np.where((formation_time  > 10) & (formation_time <= 100))

        formed_mass[i] = subparticle_masses[galexev_idx].sum() + subparticle_masses[mappings_idx].sum()

        formation_time = formation_time * 1e6

        SFRi = subparticle_masses  / (10 * 1e6) # mass/yr
        logC = np.zeros(n_resampled) + 4.5#3/5 * np.log10((subparticle_masses*u.solMass).to(u.kg) / const.M_sun) + 2/5 * np.log10(P*u.Pa / const.k_B / (u.cm**-3 * u.K))
        #print(logC)
        
        #this is the size of the HII region, its smoothing scale
        rhII = np.cbrt(10 * subparticle_masses /( np.pi/8 *density))

        #this is the radius of the sphere where the positions are to be sampled around the parent position
        hi2 = np.sqrt(np.maximum(0, (h)**2 - rhII**2))
        xn, yn, zn = sample_splined(x, y, z, hi2)
        
        xi = np.zeros(n_resampled) + x
        yi = np.zeros(n_resampled) + y
        zi = np.zeros(n_resampled) + z

        hi = np.zeros(n_resampled) + h
        Zi = np.zeros(n_resampled) + Z
        Pi = np.zeros(n_resampled) + P
        fci = np.zeros(n_resampled) + fc
        Ti = np.zeros(n_resampled) + T

        
        #['x', 'y', 'z', 'h', 'mass', 'Z', 'age']
        if len(galexev_idx[0]) > 0:
            galaxev_df = np.array([xi[galexev_idx], yi[galexev_idx], zi[galexev_idx], hi[galexev_idx], subparticle_masses[galexev_idx], Zi[galexev_idx],
                                                formation_time[galexev_idx]])
            galaxev_dfs.append(galaxev_df)
       
        if len(mappings_idx[0] > 0):

            #'x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc'
            mappings_df = np.array([xn[mappings_idx], yn[mappings_idx], zn[mappings_idx],
                                                rhII[mappings_idx], 
                                                SFRi[mappings_idx], Zi[mappings_idx],
                                                logC[mappings_idx], Pi[mappings_idx], fci[mappings_idx]])
    
            mappings_dfs.append(mappings_df)

        # particles that avoid the double counting of dust in SF regions, 3 times the area and 10 times the mass
        #all_df[['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass', 'T']
        #['x', 'y', 'z', 'h', 'mass', 'Z']
        
            ghost_particles = np.array([xn[mappings_idx], yn[mappings_idx], zn[mappings_idx], 3*rhII[mappings_idx],    
                                        -10*subparticle_masses[mappings_idx], Zi[mappings_idx],  
                                        SFRi[mappings_idx], -Ti[mappings_idx]])

            dust_ism_dfs.append(ghost_particles)
        


    remaining_content_df =  all_df.iloc[SF_particles_idx][['x', 'y', 'z', 'h', 'mass', 'Z',  'SFR', 'T']].copy().to_numpy().T
    remaining_content_df[4] -= formed_mass 
    dust_ism_dfs.append(remaining_content_df)

    if len(galaxev_dfs) > 1:
        galaxev_df = np.hstack(galaxev_dfs)
    else:
        galaxev_df = galaxev_dfs[0]

    if len(mappings_dfs) > 1:    
        mappings_df = np.hstack(mappings_dfs)
    elif len(mappings_dfs) == 1:
        mappings_df = mappings_dfs[0]
    else:
        mappings_df = []#np.array([[]*9])

    if len(dust_ism_dfs) > 1:
        dust_ism_df = np.hstack(dust_ism_dfs)
    else:
        dust_ism_df = dust_ism_dfs[0]

    #print(galaxev_df.shape[0], galaxev_df[7].sum() * u.solMass / 1e10)
    #print(mappings_df.shape[0], mappings_df[7].sum() * u.solMass / 1e10)
    #print(dust_ism_df.shape[0], dust_ism_df[7].sum() * u.solMass / 1e10)

    return galaxev_df, mappings_df, dust_ism_df


FOV = 60

if __name__ == '__main__':

    name = sys.argv[1]
    simulation = sys.argv[2]
    
    print(f'Running arepo_to_skirt for {name} in {simulation}')

    snap = int(name.split('_')[0])
    subfind = int(name.split('_')[1])
    redshift_snap = snapZ[1][snap]
    a = cosmo.scale_factor(redshift_snap)

    if not os.path.exists(f'data/{simulation}_{name}_stars_.dat'):
        print('catalogs not found, loading particle data and extracting info...')
    
        halo, json, rHalf, shPos = io(name, simulation)    
        try:
            gas = halo['PartType0']
        except:
            gas = None
        
        stellar = halo['PartType4']


        """
        Find the rotation matrix that translated the x, y plane to faceon view.
        If it is not possible to estimate it, return the identity matrix.
        """
        try:
            rotation = find_faceon_rotation(gas, stellar, shPos, rHalf, a)
            print(f'Finding face-on rotation matrix...')
            print(f'R(x, y, z) = {rotation}')
        except:
            rotation = np.identity(3)

        galaxev_df = load_particles(stellar, gas, shPos, snap, simulation)
        galaxev_df[['x', 'y', 'z', 'h', 'mass', 'Z', 'age']].to_csv(f'data/{simulation}_{name}_stars_.dat', sep=' ', float_format='%g', header=False, index=False)
    else:
        print(f'Catalog for {name} in {simulation} found. Loading...')
        galaxev_df = pd.read_csv(f'data/{simulation}_{name}_stars_.dat',  names=['x', 'y', 'z', 'h', 'mass', 'Z', 'age'], delim_whitespace=True)
        
    
    ### Plot mass map
    grid_resolution = 100 # 100 pc / pix
    galaxev_df["x_grid"] = np.floor(galaxev_df["x"] / grid_resolution).astype(int)
    galaxev_df["y_grid"] = np.floor(galaxev_df["y"] / grid_resolution).astype(int)
    galaxev_df["z_grid"] = np.floor(galaxev_df["z"] / grid_resolution).astype(int)

    mass_grid = {
    'xy' : galaxev_df.groupby(["x_grid", "y_grid"])["mass"]
                .sum()
                .unstack(fill_value=0)
                .values,
    'xz' : galaxev_df.groupby(["x_grid", "z_grid"])["mass"]
                .sum()
                .unstack(fill_value=0)
                .values,
    'yz' : galaxev_df.groupby(["y_grid", "z_grid"])["mass"]
                .sum()
                .unstack(fill_value=0)
                .values
    }

    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans,convolve
    kernel = Gaussian2DKernel(x_stddev=0.01)


    f, axs = plt.subplots(1, 3, figsize=(10, 3))


    a = axs[0].imshow(np.log10(convolve(mass_grid['xy'], kernel)), origin='lower', cmap='gist_gray')
    plt.colorbar(a, fraction=0.046, pad=0.04)
    a = axs[1].imshow(np.log10(convolve(mass_grid['xz'], kernel)), origin='lower', cmap='gist_gray')
    plt.colorbar(a, fraction=0.046, pad=0.04)
    a = axs[2].imshow(np.log10(convolve(mass_grid['yz'], kernel)), origin='lower', cmap='gist_gray')
    plt.colorbar(a, fraction=0.046, pad=0.04)
    axs[1].set_title(f'{name} ' + r'$\log \rm \Sigma_{M_\odot}$')
    plt.subplots_adjust(wspace=0.4)

    def fixticks(axs):
        tuples = [('x', 'y'), ('x', 'z'), ('y', 'z')]
        size = mass_grid['xy'].shape[0]
        MIN = FOV*1000 / grid_resolution
        for ax, t in zip(axs, tuples):
            ax.set_xticks([0, np.percentile(np.linspace(0, size, 100), 25), np.percentile(np.linspace(0, size, 100), 50), np.percentile(np.linspace(0, size, 100), 75), np.percentile(np.linspace(0, size, 100), 100)])
            ax.set_yticks([0, np.percentile(np.linspace(0, size, 100), 25), np.percentile(np.linspace(0, size, 100), 50), np.percentile(np.linspace(0, size, 100), 75), np.percentile(np.linspace(0, size, 100), 100)])
            
            ax.set_xticklabels([-30, -15, 0, 15, 30])
            ax.set_yticklabels([-30, -15, 0, 15, 30])
            ax.set_xlabel(f'{t[0]} [kpc]')
            ax.set_ylabel(f'{t[1]} [kpc]')
        
    fixticks(axs)

    plt.savefig(f'plots/{simulation}_{name}_stellarmap.png')



