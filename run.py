import time
import sys

import pandas as pd
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
from pandas import Series, DataFrame
from utils import read_argo_data, get_obj_feats, get_lane_graph
from argoverse_api.argoverse.map_representation.map_api import ArgoverseMap
import time

from xml.etree.ElementTree import Element, SubElement, ElementTree, dump
import json

MAX_PIPE_BUFFER_SIZE = 64 * 1024 * 1024
IN_PIPE_NAME = "\\\\.\\pipe\\python_in"
OUT_PIPE_NAME = "\\\\.\\pipe\\c_in"

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', type=str, default='saving')
parser.add_argument('--filter', action='store_true', default=False)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
config = parser.parse_args()
myProj = Proj(proj='utm', zone=52, ellps='WGS84', datum='WGS84', units='m')
from importlib import import_module

model = import_module('lanegcn')
config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def update_mem(mem, new, am, fov):
    time_list = mem['TIMESTAMP'].values.tolist()
    time_list_sort = []
    [time_list_sort.append(x) for x in time_list if x not in time_list_sort]
    if len(time_list_sort) == 50:
        print('prediction run')
        id_list = list(mem['TRACK_ID'])
        full_id = []
        for i in range(len(id_list)):
            if id_list.count(id_list[i]) == 50:
                if not(id_list[i] in full_id):
                    full_id.append(id_list[i])
        full_id.remove('000-0000-0000')
        init = time.time()
        data_tot = {'city': [],
                    'orig': [],
                    'gt_preds': [],
                    'has_preds': [],
                    'theta': [],
                    'rot': [],
                    'feats': [],
                    'ctrs': [],
                    'graph': []}
        for i in range(len(full_id)):
            data_save = mem.copy()

            indices = [j for j, x in enumerate(list(data_save['TRACK_ID'])) if x == full_id[i]]
            for k in indices:
                data_save.iat[k, 2] = 'AGENT'

            print(data_input['feats'])
            data_input = read_argo_data('HMC',data_save)
            print('1', data_input['feats'])
            data_input = get_obj_feats(data_input)
            print('2', data_input['feats'])
            data_input['graph'] = get_lane_graph(data_input, am)
            print('3', data_input['feats'])

            data_tot['city'].append(data_input['city'])
            data_tot['orig'].append(data_input['orig'])
            data_tot['gt_preds'].append(data_input['gt_preds'])
            data_tot['has_preds'].append(data_input['has_preds'])
            data_tot['theta'].append(data_input['theta'])
            data_tot['rot'].append(data_input['rot'])
            data_tot['feats'].append(data_input['feats'])
            data_tot['ctrs'].append(data_input['ctrs'])
            data_tot['graph'].append(data_input['graph'])
        print(data_tot["feats"])
        output = net(data_tot)

        print('conversion finished', time.time()-init)

        # new_mem = DataFrame(raw_data)
    else:
        new_mem = pd.concat([mem, new])


    return new_mem

def pipe_server():
    # region Config
    am = ArgoverseMap()
    pipe_buffer_size = ctypes.sizeof(message_t)
    msg = message_t()
    msg.id = MESSAGE_OK
    message_type = MESSAGE_OK
    data_num = 0
    # endregion
    pipe = Pipes(IN_PIPE_NAME, OUT_PIPE_NAME, MAX_PIPE_BUFFER_SIZE, connect_first=False)

    raw_data = {'TIMESTAMP': [],
                'TRACK_ID': [],
                'OBJECT_TYPE': [],
                'X': [],
                'Y': [],
                'CITY_NAME': [],
                'HEADING': []}
    mem = DataFrame(raw_data)
    while True:
        # try:
        status, resp = pipe.read_pipe(pipe_buffer_size)
        if status == 0:
            if message_type == MESSAGE_OK:
                # Read message
                msg = message_t.from_buffer_copy(resp)
                pipe_buffer_size = msg.size
                message_type = msg.id

            elif message_type == MESSAGE_OBJ:
                num_obj = int(len(resp) / 44)

                obj = dict()
                for i in range(num_obj):
                    data = resp[44 * i:44 * i + 44]
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

                    if len(veh_list) > 0 and obj_info['obj_id'] != 0:
                        cur_time = veh_list[0][0]
                        id_mask = '000-0000-0000'
                        obj_id = id_mask[:-len(str(obj_info['obj_id']))] + str(obj_info['obj_id'])
                        x = veh_list[0][3] + obj_info['rel_pos_long'] * np.cos(np.deg2rad(veh_list[0][6])) - obj_info['rel_pos_lat'] * np.sin(np.deg2rad(veh_list[0][6]))
                        y = veh_list[0][4] + obj_info['rel_pos_long'] * np.sin(np.deg2rad(veh_list[0][6])) + obj_info['rel_pos_lat'] * np.cos(np.deg2rad(veh_list[0][6]))
                        yaw = np.mod(veh_list[0][6] + obj_info['heading_angle'],360)
                        veh_list.append([cur_time, obj_id, 'OTHERS', x, y, 'HMC', yaw])
                    data_temp = pd.DataFrame(veh_list, columns=['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME', 'HEADING'])
                if data_num > 1000:
                    print('mem')
                    mem = update_mem(mem, data_temp, am, 100)

                # Send OK Message
                msg.id = MESSAGE_OK
                pipe.write_pipe(ctypes.string_at(ctypes.byref(msg), ctypes.sizeof(msg)))
                pipe_buffer_size = ctypes.sizeof(message_t)

                # Set message type to OK
                message_type = MESSAGE_OK
            elif message_type == MESSAGE_EGO:
                veh_list = []
                ego = dict()
                ego['time_stamp'] = int.from_bytes(resp[0:8], 'little', signed=False)
                ego['speed_mps'] = struct.unpack('<d', resp[8:16])[0]
                ego['lat_deg'] = struct.unpack('<d', resp[16:24])[0]
                ego['long_deg'] = struct.unpack('<d', resp[24:32])[0]
                ego['heading'] = struct.unpack('<f', resp[32:36])[0]
                ego['X'], ego['Y'] = myProj(ego['long_deg'], ego['lat_deg'])
                if len(veh_list) == 0:
                    veh_list.append([ego['time_stamp']* 0.001, '000-0000-0000', 'AV', ego['X'], ego['Y'], 'HMC', ego['heading']])
                cur_time = ego['time_stamp'] * 0.001

                # Send OK Message
                msg.id = MESSAGE_OK
                pipe.write_pipe(ctypes.string_at(ctypes.byref(msg), ctypes.sizeof(msg)))
                pipe_buffer_size = ctypes.sizeof(message_t)
                data_num = data_num + 1
                # Set message type to OK
                message_type = MESSAGE_OK


pipe_server()
