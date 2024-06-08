import numpy as np
import torch

def noise(a, power):
    mean = 10
    std_dev = 0
    if power == 'Low':
        std_dev = 10
    elif power == 'Middle':
        std_dev = 20
    elif power == 'High':
        std_dev = 30
    elif power == 'Extra':
        std_dev = 40
    noise = np.random.normal(mean, std_dev, a.shape).astype(np.uint8)
    return a+noise

def sheltering(a, power):
    b = a.copy()
    w, h = a.shape[0], a.shape[1]
    mid = a[w // 2 - 20 : w // 2 + 20, h // 2 - 20 : h // 2 + 20]
    if power == 'Low':
        b[40:80, 40:80] = mid
    elif power == 'Middle':
        b[40:80, 40:80] = mid
        b[-80:-40, -80:-40] = mid
    elif power == 'High':
        b[40:80, 40:80] = mid
        b[-80:-40, -80:-40] = mid
        b[40:80, -80:-40] = mid
        b[-80:-40, 40:80] = mid
    elif power == 'Extra':
        b[40:80, 40:80] = mid
        b[-80:-40, -80:-40] = mid
        b[40:80, -80:-40] = mid
        b[-80:-40, 40:80] = mid
        b[w // 2 - 20 : w // 2 + 20, 40:80] = mid
        b[w // 2 - 20 : w // 2 + 20, -80:-40] = mid
        b[40:80, h // 2 - 20 : h // 2 + 20] = mid
        b[-80:-40, h // 2 - 20 : h // 2 + 20] = mid
    return b
    
def exchange(a, power):
    b = a.copy()
    if power == 'Low':
        f1 = a[40:80, 40:80]
        f2 = a[-80:-40, -80:-40]
        b[40:80, 40:80] = f2
        b[-80:-40, -80:-40] = f1
    elif power == 'Middle':
        f1 = a[40:80, 40:80]
        f2 = a[-80:-40, -80:-40]
        f3 = a[40:80, -80:-40]
        b[40:80, -80:-40] = f1
        b[40:80, 40:80] = f2
        b[-80:-40, -80:-40] = f3
    elif power == 'High':
        f1 = a[40:80, 40:80]
        f2 = a[-80:-40, -80:-40]
        f3 = a[40:80, -80:-40]
        f4 = a[-80:-40, 40:80]
        b[40:80, 40:80] = f4
        b[-80:-40, -80:-40] = f3
        b[40:80, -80:-40] = f2
        b[-80:-40, 40:80] = f1
    elif power == 'Extra':
        w, h = a.shape[0], a.shape[1]
        f1 = a[40:80, 40:80]
        f2 = a[-80:-40, -80:-40]
        f3 = a[40:80, -80:-40]
        f4 = a[-80:-40, 40:80]
        f5 = a[w // 2 - 20 : w // 2 + 20, h // 2 - 20 : h // 2 + 20]
        b[w // 2 - 20 : w // 2 + 20, h // 2 - 20 : h // 2 + 20] = f1
        b[40:80, 40:80] = f2
        b[-80:-40, -80:-40] = f3
        b[40:80, -80:-40] = f4
        b[-80:-40, 40:80] = f5
    return b