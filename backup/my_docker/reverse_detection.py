
import matplotlib.pyplot as plt
import glob
import numpy as np
import re 
import neurokit2 as nk
import time 
from tqdm import tqdm
import pandas as pd
from scipy.signal import medfilt

def rri_ver_2(signal, sr=250):
    peaks, _ = nk.ecg_peaks(signal, sampling_rate=sr)
    r_peaks = peaks[peaks["ECG_R_Peaks"] == 1].index.tolist()
    rr_interval = np.diff(r_peaks)
    r_peaks.pop(0)
    return np.array(r_peaks), np.average(rr_interval)

def p_detection(temp):
    q_peak = np.min(temp[int(len(temp)*0.4)-10:int(len(temp)*0.4)])
    q_index = np.where(temp == q_peak)[0][0]
    p_min = np.min(temp[20:q_index-10])
    p_max = np.max(temp[20:q_index-10])
    return q_peak, q_index, p_min, p_max

def reverse_caculate(p_min, p_max):
    p_min_abs = np.abs(p_min)
    p_max_abs = np.abs(p_max)
    title = 0
    if p_min_abs >= p_max_abs:
        title = 1
    return title

def median_filter(signal):
    xx1 = medfilt(signal, 29)
    baseline = medfilt(xx1, 99)
    return signal - baseline

def plot_pq(id, reverse, pred_reverse, temp, q, p_min, p_max, t_max, t_min):
    plt.title(f"{id}/{pred_reverse}")
    plt.plot(temp)
    plt.scatter(q+40, temp[q+40], color="red", label="q")
    plt.scatter(p_min+30, temp[p_min+30], color="purple", label="p min")
    plt.scatter(p_max+30, temp[p_max+30], color="yellow", label="p max")
    plt.scatter(t_max+120, temp[t_max+120],color="blue", label="t_max")
    plt.scatter(t_min+120, temp[t_min+120], color="darkkhaki", label="t_min")
    plt.legend()
    plt.show()

