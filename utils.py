import os
import sys
import h5py 
import requests
import json as j
import numpy as np
import pandas as pd

from orientation import *

from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
from sklearn.neighbors import KDTree

h = 0.6774
cosmo = LambdaCDM(H0=100*h, Om0=0.3089, Ob0=0.0486, Ode0=0.6911, Tcmb0=2.73)

snapZ = np.load('snapTNG.npy')
headers = {"api-key":"ff352a2affacf64753689dd603b5b44e"} #replace with your APIKEY


def rndm(a, b, g, size=1):

    def power_law(k_min, k_max, y, gamma):
        return ((k_max**(-gamma+1) - k_min**(-gamma+1))*y  + k_min**(-gamma+1.0))**(1.0/(-gamma + 1.0))

    scale_free_distribution = np.zeros(size, float)
    gamma = 1.8

    for n in range(size):
        scale_free_distribution[n] = power_law(a, b, np.random.uniform(0,1), gamma)
    
    return scale_free_distribution

def get(path, filename=None, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    
    if 'content-disposition' in r.headers:
        print(r.headers['content-disposition'].split("filename=")[1])
        with open(f'particledata/{filename}.hdf5', 'wb') as f:
            f.write(r.content)

        return filename # return the filename string
    
    return r

def verify_if_already_downloaded(name):

    cutout = os.path.exists(f'/home/ppxlf2/TNG_cutouts/halo_{name}.hdf5')
    json = os.path.exists(f'/home/ppxlf2/TNG_cutouts/{name}.json')
    stars = os.path.exists(f'/home/ppxlf2/skirts_data/{name}_stars.dat')
    starbursting = os.path.exists(f'/home/ppxlf2/skirts_data/{name}_starbursting.dat')
    dust = os.path.exists(f'/home/ppxlf2/skirts_data/{name}_dust.dat')

    return cutout & json & stars & starbursting & dust


def smoothing_length(x, y, z, reference=None, Nk=64, return_index=False):
    X = np.vstack([x, y, z]).T
  
    if reference is not None:
        tree = KDTree(reference, leaf_size=2)
    else:
        tree = KDTree(X, leaf_size=2)

    dist, ind = tree.query(X, k=Nk) 

    if return_index:
        return dist[:,-1], ind
    else:
        return dist[:,-1]

def io(name, simulation='TNG50-1', halomode=True):

    snap = int(name.split('_')[0])
    subfind = int(name.split('_')[1])

    url = f'https://www.tng-project.org/api/{simulation}/snapshots/{snap}/subhalos/{subfind}/'

    json = get(url)
        
    rHalf = json['halfmassrad_gas'] #* a / cosmo.h
    shPos = np.array([json['pos_x'], json['pos_y'], json['pos_z']]) #* a / cosmo.h
    
    if halomode:
        print('Working on halo mode, all halo particles are going to be used.')
        halo_url = json['cutouts']['parent_halo']
        halo_id = halo_url.split('/')[-2]
        filename = f'cutout_{simulation}_{snap}_{subfind}_{halo_id}'
        halo_url = halo_url.replace('http:', 'https:')

        if not os.path.exists(f'particledata/{filename}.hdf5'):
            print(f'Particle data not found. Downloading...')
            filename = get(halo_url, filename=f'{filename}')
        else:
            print(f'Particle data found. Using {filename}.hdf5')

    else:
        print('Working on subhalo mode, only subhalo particles are going to be used.')
        halo_url = json['cutouts']['subhalo']
        halo_id = halo_url.split('/')[-2]
        filename = f'cutout_{simulation}_{snap}_{subfind}_{halo_id}'
        halo_url = halo_url.replace('http:', 'https:')
        if not os.path.exists(f'particledata/{filename}.hdf5'):
            print(f'Particle data not found. Downloading...')
            filename = get(halo_url, filename=f'{filename}')
        else:
            print(f'Particle data found. Using {filename}.hdf5')

    
    

    if halomode:
        halo = h5py.File(f'particledata/{filename}.hdf5', 'r')
    else:
        halo = h5py.File(f'particledata/{filename}.hdf5', 'r')

    return halo, json, rHalf, shPos
