from arena_api.system import system
from arena_api.buffer import *

import ctypes
import numpy as np
import cv2
import time

def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			print(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			print(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')

def dop(i1, i2, i3, i4):
    return np.sqrt((i1-i3)**2+(i2-i4)**2)/(i1+i2+i3+i4+1e-8)*2

def dop_img(i1, i2, i3, i4):
	dop = np.sqrt((i1-i3)**2+(i2-i4)**2)/(i1+i2+i3+i4+1e-8)*2
	img = (dop.clip(0,1)*255).astype(np.uint8)
	return img

def aop(i1, i2, i3, i4):
    return 0.5*np.arctan2((i2-i4),(i1-i3+1e-8))

def aop_img(i1, i2, i3, i4):
	aop = 0.5*np.arctan2((i2-i4),(i1-i3+1e-8))
	img = (((aop/np.pi*2 + 1)/2.).clip(0, 1)*255).astype(np.uint8)
	return img

def get_unpol_img(i1, i2, i3, i4):
	i_unpol = (i1 + i2 + i3 + i4) / 4 * 255
	img = i_unpol.clip(0,255).astype(np.uint8)
	if len(img.shape) == 2:
		img = img[..., None].repeat(3, axis=-1)
	return img 
	