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
    
    nodes['ExposureTime'].value = min(args.exposure_time, nodes['ExposureTime'].max)
    args.exposure_time = nodes['ExposureTime'].value
    nodes['BalanceWhiteEnable'].value = True
    nodes['BalanceWhiteAuto'].value = 'Continuous'
    # Nodes
    # print('Setting Width to 1224')
    nodes['Width'].value = 1224

    # print('Setting Height to 1024')
    height = nodes['Height']
    height.value = 1024

    # Set pixel format to Mono8, most cameras should support this pixel format
    pixel_format_name = 'PolarizedAngles_0d_45d_90d_135d_BayerRG8'
    # print(f'Setting Pixel Format to {pixel_format_name}')
    nodes['PixelFormat'].value = pixel_format_name
    if args.debug:
        print('Settings:')
        for key in nodes:
            print(f'\t{key:20s} ==> {nodes[key].value}')

    # Grab and save an image buffer -------------------------------------------
    print('Starting stream')
    with device.start_stream(1):
        stream_flag = True
        
        # print('Grabbing an image buffer')
        # Optional args
        image_buffer = device.get_buffer()
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
        image_only_data = None
        if image_buffer.has_chunkdata:
            # 8 is the number of bits in a byte
            bytes_pre_pixel = int(image_buffer.bits_per_pixel / 8)

            image_size_in_bytes = image_buffer.height * \
                image_buffer.width * bytes_pre_pixel

            image_only_data = image_buffer.data[:image_size_in_bytes]
        else:
            image_only_data = image_buffer.data

        nparray = np.asarray(image_only_data, dtype=np.uint8)

        # Reshape array for pillow
        nparray_reshaped = nparray.reshape((
            image_buffer.height,
            image_buffer.width,
            4
        ))

        #
        # Method 2 (from Buffer.pdata)
        #
        # A more general way (not used in this simple example)
        #
        # Creates an already reshaped array to use directly with
        # pillow.
        # np.ctypeslib.as_array() detects that Buffer.pdata is (uint8, c_ubyte)
        # type so it interprets each byte as an element.
        # For 16Bit images Buffer.pdata must be cast to (uint16, c_ushort)
        # using ctypes.cast(). After casting, np.ctypeslib.as_array() can
        # interpret every two bytes as one array element (a pixel).
        #
        # Code:
        '''
        nparray_reshaped = np.ctypeslib.as_array(
        image_buffer.pdata,
        (image_buffer.height, image_buffer.width))
        '''
        # Save in args.display_dir
        # png_array = cv2.cvtColor(nparray_reshaped[..., 0], cv2.COLOR_BAYER_BG2BGR)
        # cv2.imwrite(os.path.join(args.display_dir, 'display.png'), png_array)

        # Save image
        # print('Saving image')
        exp_time_int = int(args.exposure_time/1000)
        for i in range(4):
            png_name = f'{args.scene_num:s}_{args.capture_num:03d}_{exp_time_int:03d}ms_{i*45:03d}.png'
            png_array = cv2.cvtColor(nparray_reshaped[..., i], cv2.COLOR_BAYER_BG2BGR)
            cv2.imwrite(os.path.join(args.save_dir, png_name), png_array, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        print(f'Saved image to: {os.path.join(args.save_dir, png_name)}')
        txt_name = png_name.replace('_135.png', '.log')
        with open(os.path.join(args.save_dir, txt_name), 'w') as f:
            f.write(f'exposure time: {args.exposure_time}us')

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
    parser.add_argument('--exposure_ratio', '-e', default=1, type=float)
    parser.add_argument('--save_dir', default='C:\\Users\\pkuCamLab\\Desktop\\new_capture', type=str)
    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    args.exposure_time = args.exposure_ratio * args.exposure_base
    # args.display_dir = os.path.join(args.save_dir, 'display')
    # if not os.path.isdir(args.display_dir):
    #     os.makedirs(args.display_dir)
    # print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
    # print('\nExample started\n')
	
    example_entry_point(args)
    # print('\nExample finished successfully')
