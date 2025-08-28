import os

import matplotlib
import numpy as np
import rerun as rr
import json
from tqdm import tqdm
from scipy.stats import multivariate_normal



def medianTrajError(output, target):

    diff = np.linalg.norm(target - output, axis = 1)
    orderedDiff = np.sort(diff)

    return orderedDiff[len(orderedDiff)//2]


def averageTrajError(output, target):

    diff = np.linalg.norm(target - output, axis = 1)

    return np.mean(diff, axis = 0)


def pointTrack(queryPoint, anchorPos, anchorRot):
    R = qToRot(anchorRot[0])
    
    t0 = R.T@(queryPoint - anchorPos[0])
    track = []
    for idx in tqdm(range(len(anchorPos)), 'Track', position = 1, leave = False):
        track.append(anchorPos[idx] + qToRot(anchorRot[idx])@t0)
    
    return np.array(track)

def qToRot(q):
    norm = np.linalg.norm(q)
    r = q[0]/norm
    x = q[1]/norm
    y = q[2]/norm
    z = q[3]/norm


    R = np.array(
        [[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)],
        [2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)],
        [2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)]]
    )
    return R

def get3DCov(scale, rotation, scale_mod = 1):
    
    S = np.zeros((3,3))
    S[0][0] = scale_mod * scale[0]
    S[1][1] = scale_mod * scale[1]
    S[2][2] = scale_mod * scale[2]

    R = qToRot(rotation)
    M = S * R
    
    sigma = np.transpose(M) * M
    
    return sigma

def getAll3DCov(scales, rotations, scale_mod = 1):

    cov3Ds = []
    for idx in tqdm(range(len(scales)), 'Cov'):
        cov3Ds.append(get3DCov(scales[idx], rotations[idx], scale_mod))

    return np.array(cov3Ds)

def getContributions(mean3Ds, cov3Ds, query):

    assert len(mean3Ds) == len(cov3Ds), f'{mean3Ds.shape} {cov3Ds.shape}'

    PDFs = []

    for idx in tqdm(range(len(mean3Ds)),'PDF', position = 1, leave = False):
        try:
            pdf = multivariate_normal.pdf(query, mean = mean3Ds[idx], cov = cov3Ds[idx])
            PDFs.append(pdf)
        except:
            PDFs.append(-1)

    return np.array(PDFs)
    

