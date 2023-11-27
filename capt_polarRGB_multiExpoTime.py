# -----------------------------------------------------------------------------
# Copyright (c) 2021, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

import os  # os.getcwd()
import time
import argparse
import ctypes

import cv2
import numpy as np  # pip install numpy

import arena_utils as autil

from arena_api.system import system

def example_entry_point(args):
    # Create a device
    devices = autil.create_devices_with_tries()
    device = devices[0]
    # Get device stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap
    # Enable stream auto negotiate packet size
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

    # Enable stream packet resend
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    # Get/Set nodes -----------------------------------------------------------
    nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat', 
                                     'ExposureTime', 'ExposureAuto',
                                     'Gain', 'GainAuto',
                                     'BalanceWhiteAuto', 'BalanceWhiteEnable'])
    nodes['ExposureAuto'].value = 'Off'
    nodes['GainAuto'].value = 'Off'
    nodes['Gain'].value = 0.
    
    nodes['BalanceWhiteEnable'].value = True
    nodes['BalanceWhiteAuto'].value = 'Continuous'
    # Nodes
    # print('Setting Width to 1224')
    nodes['PixelFormat'].value = 'BayerRG16'
    nodes['Width'].value = 2448
    height = nodes['Height']
    height.value = 2048


    if args.debug:
        print('Settings:')
        for key in nodes:
            print(f'\t{key:20s} ==> {nodes[key].value}')

    num_imgs = len(args.exposure_ratios)
    exp_time_max = nodes['ExposureTime'].max

    # Grab and save an image buffer -------------------------------------------
    print(f'Starting stream, capture {num_imgs} images')
    for i in range(num_imgs):
        with device.start_stream(1):
            exp_time = args.exposure_ratios[i] * args.exposure_base
            if exp_time > exp_time_max:
                print(f'\tWarning: set exposure time {exp_time} exceed the Max value {exp_time_max}')


            exp_time = min(exp_time, exp_time_max)
            nodes['ExposureTime'].value = exp_time
            print(f'\t capt img at {int(nodes["ExposureTime"].value):d} us')
            # print('Grabbing an image buffer')
            # Optional args
            image_buffer = device.get_buffer()
            exp_time_int = int(exp_time/1000)

            # print(f' Width X Height = '
            #       f'{image_buffer.width} x {image_buffer.height}')

            # To save an image Pillow needs an array that is shaped to
            # (height, width). In order to obtain such an array we use numpy
            # library
            # print('Converting image buffer to a numpy array')

            # Buffer.pdata is a (uint8, ctypes.c_ubyte)
            # Buffer.data is a list of elements each represents one byte. Therefore
            # for Mono8 each element represents a pixel.

            #
            # Method 1 (from Buffer.data)
            #
            # dtype is uint8 because Buffer.data returns a list or bytes and pixel
            # format is also Mono8.
            # NOTE:
            # if 'ChunkModeActive' node value is True then the Buffer.data is
            # a list of (image data + the chunkdata) so data list needs to be
            # truncated to have image data only.
            # can use either :
            #  - device.nodemap['ChunkModeActive'].value   (expensive)
            #  - buffer.has_chunkdata                 (less expensive)
            pdata_as16 = ctypes.cast(image_buffer.pdata,
                                    ctypes.POINTER(ctypes.c_ushort))
            raw_data = np.ctypeslib.as_array(
                pdata_as16,
                (image_buffer.height, image_buffer.width)
            )
            # Save in args.display_dir
            pol000, pol135 = raw_data[::2,::2], raw_data[::2, 1::2]
            pol045, pol090 = raw_data[1::2,::2], raw_data[1::2, 1::2]
            unpol = ((pol090 + pol045 + pol135 + pol000) / 4.).astype(np.uint16)
            unpol_bgr = cv2.cvtColor(unpol, cv2.COLOR_BAYER_BG2BGR)
            jpg_name = f'{args.scene_num:s}_{args.capture_num:02d}_{exp_time_int:02d}ms_unpol.jpg'
            unpol_bgr = (((unpol_bgr/65535.) ** (1/2.2)).clip(0,1)*255).astype(np.uint8)
            unpol_save_path = os.path.join(args.save_dir, 'jpg', jpg_name)
            cv2.imwrite(unpol_save_path, unpol_bgr)
            # pol_arr = np.stack([pol000, pol045, pol090, pol135])
            
            png_name = f'{args.scene_num:s}_{args.capture_num:02d}_{exp_time_int:02d}ms.png'
            raw_save_path = os.path.join(args.save_dir, 'raw', png_name)
            cv2.imwrite(raw_save_path, raw_data, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            print(f'Saved image to: {raw_save_path}')
            txt_name = png_name.replace('.png', '.log')
            with open(os.path.join(args.save_dir, 'raw', txt_name), 'w') as f:
                f.write(f'exposure time: {exp_time}us')

            # time.sleep(0.1)
            device.requeue_buffer(image_buffer)

    # Clean up ---------------------------------------------------------------

    # Stop stream and destroy device. This call is optional and will
    # automatically be called for any remaining devices when the system module
    # is unloading.
    system.destroy_device()
    # print('Destroyed all created devices')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--scene_num', '-s', type=str)
    parser.add_argument('--capture_num', '-c', default=0, type=int)
    parser.add_argument('--exposure_base', default=1000., type=float)
    parser.add_argument('--exposure_ratios', '-e', default=10., type=float)
    parser.add_argument('--save_dir', default='D:\\yan\\Arena\\Arena_test\\data', type=str)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.save_dir, 'jpg'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'raw'), exist_ok=True)
    
    # args.display_dir = os.path.join(args.save_dir, 'display')
    # if not os.path.isdir(args.display_dir):
    #     os.makedirs(args.display_dir)
    # print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
    # print('\nExample started\n')
    args.exposure_ratios = [args.exposure_ratios]
    example_entry_point(args)
    # print('\nExample finished successfully')
