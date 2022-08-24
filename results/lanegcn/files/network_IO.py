import time
import sys
import win32pipe, win32file, pywintypes
from lib.utils import *
import struct
import math
import argparse
import pickle
import glob
from pyproj import Proj
import os.path
from os import path

MAX_PIPE_BUFFER_SIZE = 64*1024*1024
IN_PIPE_NAME = "\\\\.\\pipe\\python_in"
OUT_PIPE_NAME = "\\\\.\\pipe\\c_in"

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', type=str, default='saving')
parser.add_argument('--filter', action='store_true', default=False)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
config = parser.parse_args()
myProj = Proj(proj='utm', zone=52, ellps='WGS84', datum='WGS84', units='m')

def pipe_server():
    #region Config
    pipe_buffer_size = ctypes.sizeof(message_t)
    msg = message_t();
    msg.id = MESSAGE_OK
    message_type = MESSAGE_OK

    #endregion
    pipe = Pipes(IN_PIPE_NAME, OUT_PIPE_NAME, MAX_PIPE_BUFFER_SIZE, connect_first=False)
    print("pipe server")
    
    try:
        while True:
            status, resp = pipe.read_pipe(pipe_buffer_size)
            if status==0:
                print("Status:", status)
                if message_type==MESSAGE_OK:
                    # Read message
                    msg = message_t.from_buffer_copy(resp)
                    print("Message: %d"%msg.id)
                    pipe_buffer_size = msg.size
                    message_type = msg.id

                elif message_type==MESSAGE_OBJ:
                    print("OBJ received")
                    num_obj = int(len(resp)/44)
                    print(num_obj)

                    obj = dict()
                    for i in range(num_obj):
                        data = resp[44*i:44*i+44]
                        if i == 0:
                            obj['time_stamp'] = int.from_bytes(data[0:4], 'little', signed=False)
                            obj['obj_num'] = num_obj
                            obj['obj_infos'] = []

                            obj_info = dict()
                            obj_info['obj_id'] = int.from_bytes(data[4:8], 'little', signed=False)
                            obj_info['rel_pos_lat'] = struct.unpack('<f', data[8:12])[0]
                            obj_info['rel_pos_long'] = struct.unpack('<f', data[12:16])[0]
                            obj_info['rel_pos_height'] = struct.unpack('<f', data[16:20])[0]
                            obj_info['heading_angle'] = struct.unpack('<f', data[20:24])[0]
                            obj_info['rel_vel_lat'] = struct.unpack('<f', data[24:28])[0]
                            obj_info['rel_vel_long'] = struct.unpack('<f', data[28:32])[0]
                            obj_info['Abs_speed'] = struct.unpack('<f', data[32:36])[0]
                            obj_info['obj_width'] = struct.unpack('<f', data[36:40])[0]
                            obj_info['obj_length'] = struct.unpack('<f', data[40:44])[0]
                            obj['obj_infos'].append(obj_info)
                        else:
                            obj_info = dict()
                            obj_info['obj_id'] = int.from_bytes(data[4:8], 'little', signed=False)
                            obj_info['rel_pos_lat'] = struct.unpack('<f', data[8:12])[0]
                            obj_info['rel_pos_long'] = struct.unpack('<f', data[12:16])[0]
                            obj_info['rel_pos_height'] = struct.unpack('<f', data[16:20])[0]
                            obj_info['heading_angle'] = struct.unpack('<f', data[20:24])[0]
                            obj_info['rel_vel_lat'] = struct.unpack('<f', data[24:28])[0]
                            obj_info['rel_vel_long'] = struct.unpack('<f', data[28:32])[0]
                            obj_info['Abs_speed'] = struct.unpack('<f', data[32:36])[0]
                            obj_info['obj_width'] = struct.unpack('<f', data[36:40])[0]
                            obj_info['obj_length'] = struct.unpack('<f', data[40:44])[0]
                            obj['obj_infos'].append(obj_info)
                    cur_time = obj['time_stamp']
                    file_name = 'saving/objs/' + str(cur_time) + '.pkl'
                    with open(file_name, 'wb') as f:
                        pickle.dump(obj, f)

                    # Send OK Message
                    msg.id = MESSAGE_OK
                    pipe.write_pipe(ctypes.string_at(ctypes.byref(msg), ctypes.sizeof(msg)))
                    print("Message Sent!")
                    pipe_buffer_size = ctypes.sizeof(message_t)

                    # Set message type to OK
                    message_type = MESSAGE_OK
                elif message_type==MESSAGE_EGO:
                    print("EGO received")

                    ego = dict()

                    ego['time_stamp'] = int.from_bytes(resp[0:8], 'little', signed=False)
                    ego['speed_mps'] = struct.unpack('<d', resp[8:16])[0]
                    ego['lat_deg'] = struct.unpack('<d', resp[16:24])[0]
                    ego['long_deg'] = struct.unpack('<d', resp[24:32])[0]
                    ego['heading'] = struct.unpack('<f', resp[32:36])[0]
                    print(ego)
                    cur_time = ego['time_stamp']
                    file_name = 'saving/ego/' + str(cur_time) + '.pkl'
                    with open(file_name, 'wb') as f:
                        pickle.dump(ego, f)

                    # Send OK Message
                    msg.id = MESSAGE_OK
                    pipe.write_pipe(ctypes.string_at(ctypes.byref(msg), ctypes.sizeof(msg)))
                    print("Message Sent!")
                    pipe_buffer_size = ctypes.sizeof(message_t)

                    # Set message type to OK
                    message_type = MESSAGE_OK

                elif message_type==MESSAGE_MAP:
                    # with open("map.data", "wb") as f:
                    #     f.write(resp)
                    
                    print("Map received")
                    num_lane_links = int(len(resp)/9024)
                    for i in range(num_lane_links):
                        data = resp[9024*i:9024*i+9024]
                        lane_link = dict()
                        lane_link['next_ids_count'] = int.from_bytes(data[0:4], 'little', signed=False)
                        lane_link['prev_ids_count'] = int.from_bytes(data[88:92], 'little', signed=False)
                        lane_link['next_id'] = []
                        lane_link['prev_id'] = []
                        for num_cnt in range(10):
                            lane_link['next_id'].append(int.from_bytes(data[8*num_cnt+8:8*num_cnt+16], 'little', signed=False))
                            lane_link['prev_id'].append(int.from_bytes(data[8 * num_cnt + 96:8 * num_cnt + 104], 'little', signed=False))
                        lane_link['ref_lat'] = struct.unpack('<d', data[176:184])[0]
                        lane_link['ref_lng'] = struct.unpack('<d',data[184:192])[0]
                        ref_utm_x, ref_utm_y = myProj(lane_link['ref_lng'], lane_link['ref_lat'])

                        lane_link['points_count'] = int.from_bytes(data[192:196], 'little', signed=False)
                        lane_link['points_x'] = []
                        lane_link['points_y'] = []
                        lane_link['points_x_utm'] = []
                        lane_link['points_y_utm'] = []
                        for num_cnt in range(550):
                            lane_link['points_x'].append(struct.unpack('<d', data[8 * num_cnt + 200:8 * num_cnt + 208])[0])
                            lane_link['points_y'].append(struct.unpack('<d', data[8 * num_cnt + 4600:8 * num_cnt + 4608])[0])
                            lane_link['points_x_utm'].append(struct.unpack('<d', data[8*num_cnt+200:8*num_cnt+208])[0]+ref_utm_x)
                            lane_link['points_y_utm'].append(struct.unpack('<d', data[8*num_cnt+4600:8*num_cnt+4608])[0]+ref_utm_y)
                        lane_link['id']=int.from_bytes(data[9000:9008], 'little', signed=False)
                        lane_link['left_id'] = int.from_bytes(data[9008:9016], 'little', signed=False)
                        lane_link['right_id'] = int.from_bytes(data[9016:9024], 'little', signed=False)

                        file_name = 'saving/map/' + str(lane_link['id']) + '.pkl'
                        if path.exists(file_name)==True:
                            pass
                        else:
                            with open(file_name, 'wb') as f:
                                pickle.dump(lane_link, f)

                    print(num_lane_links)
                    # Send OK Message
                    msg.id = MESSAGE_OK
                    pipe.write_pipe(ctypes.string_at(ctypes.byref(msg), ctypes.sizeof(msg)))
                    print("Message Sent!")
                    pipe_buffer_size = ctypes.sizeof(message_t)
                    
                    # Set message type to OK
                    message_type = MESSAGE_OK


        print("finished now")
    finally:
        win32file.CloseHandle(pipe)

pipe_server()

## 질문사항
## points_count 갯수 이상으로 데이터가 나올수 있는지 (buffer초기화가 안되서 이전거 물고 있던 것 같기도)
## ref lat이랑 ref lng이 모두 동일하게 나오는 것 같은데 이게 맞는지
##