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


def update_mem(mem, new, data_count):
    time_list = mem['TIMESTAMP'].values.tolist()
    time_list_sort = []
    [time_list_sort.append(x) for x in time_list if x not in time_list_sort]
    print(len(time_list_sort))
    if len(time_list_sort) == 50:
        print('save')
        cls_rand = np.random.rand()
        if cls_rand < 0.15:
            cls = 'test_obs'
        elif cls_rand < 0.3:
            cls = 'val'
        else:
            cls = 'train'
        print(mem)
        data_count = data_count + 1
        mem.to_csv('D://research//trajectory_prediction//dataset//HMC//'+cls+'//data//'+str(data_count)+'.csv', index=False)
        raw_data = {'TIMESTAMP': [],
                    'TRACK_ID': [],
                    'OBJECT_TYPE': [],
                    'X': [],
                    'Y': [],
                    'CITY_NAME': [],
                    'HEADING': []}
        new_mem = DataFrame(raw_data)
    else:
        new_mem = pd.concat([mem, new])


    return new_mem, data_count

def map_conversion():
    root = Element('ArgoverseVectorMap')
    lane_id_map = dict()
    link_list = glob.glob('saving/map/*.pkl')
    halluc_bbox = np.empty(shape=(len(link_list), 4))

    for i in range(len(link_list)):
        with open(link_list[i], 'rb') as f:
            data = pickle.load(f)
        if i == 0:
            x = data['points_x_utm'][:data['points_count']]
            y = data['points_y_utm'][:data['points_count']]
            pt_list = np.array([x,y]).T
        else:
            x = data['points_x_utm'][:data['points_count']]
            y = data['points_y_utm'][:data['points_count']]
            pt_list = np.concatenate((pt_list, np.array([x,y]).T), axis=0)
        halluc_bbox[i, 0] = np.min(x)
        halluc_bbox[i, 1] = np.min(y)
        halluc_bbox[i, 2] = np.max(x)
        halluc_bbox[i, 3] = np.max(y)

    [_, indexes] = np.unique(pt_list, axis=0, return_index=True)
    wps = pt_list[sorted(indexes)]

    for i in range(len(wps)):
        id = i
        x = wps[i, 0]
        y = wps[i, 1]
        SubElement(root, 'node').attrib = {"id": str(id), "x": str(x), "y": str(y)}

    for i in range(len(link_list)):
        with open(link_list[i], 'rb') as f:
            data = pickle.load(f)
        lane_id = data['id']
        lane_id_map[str(i)] = lane_id
        way = SubElement(root, 'way', lane_id=str(lane_id))
        SubElement(way, 'tag').attrib = {"k": "has_traffic_control", "v": "False"}
        SubElement(way, 'tag').attrib = {"k": "turn_direction", "v": "NONE"}
        SubElement(way, 'tag').attrib = {"k": "is_intersection", "v": "False"}
        SubElement(way, 'tag').attrib = {"k": "l_neighbor_id", "v": "None"}
        SubElement(way, 'tag').attrib = {"k": "r_neighbor_id", "v": "None"}
        x = data['points_x_utm'][:data['points_count']]
        y = data['points_y_utm'][:data['points_count']]
        points_in_link = np.array([x, y]).T
        for j in range(len(points_in_link)):
            wps_pt = np.array([points_in_link[j][0], points_in_link[j][1]])
            for z in range(len(wps)):
                wps_cand = wps[z]
                if wps_cand[0] == wps_pt[0] and wps_cand[1] == wps_pt[1]:
                    SubElement(way, 'nd').attrib = {"ref": str(z)}
        if data['prev_ids_count'] == 0:
            SubElement(way, 'tag').attrib = {"k": "predecessor", "v": "None"}
        else:
            for j in range(data['prev_ids_count']):
                pred_id = data['prev_id'][j]
                SubElement(way, 'tag').attrib = {"k": "predecessor", "v": str(pred_id)}
        if data['next_ids_count'] == 0:
            SubElement(way, 'tag').attrib = {"k": "successor", "v": "None"}
        else:
            for j in range(data['next_ids_count']):
                suc_id = data['next_id'][j]
                SubElement(way, 'tag').attrib = {"k": "successor", "v": str(suc_id)}

    indent(root)
    dump(root)
    tree = ElementTree(root)

    xmin = np.min(wps[:, 0])
    xmax = np.max(wps[:, 0])
    ymin = np.min(wps[:, 1])
    ymax = np.max(wps[:, 1])
    grid = np.zeros(shape=(int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))))
    grid_x = np.zeros(shape=(int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))))
    grid_y = np.zeros(shape=(int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))))
    drivable_region = np.zeros(shape=(int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))), dtype=np.uint8)
    ground_height = np.empty(shape=(int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))), dtype=np.float32)
    ground_height[:] = np.nan
    city_se2 = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    map = 'Suburb_02'
    map_abbr = 'HMC'
    map_id = '10317'

    tree.write('dataset/HMC/map_files/pruned_argoverse_' + map_abbr + '_' + map_id + '_vector_map.xml', encoding='utf-8', xml_declaration=True)
    with open('dataset/HMC/map_files/' + map_abbr + '_' + map_id + "_tableidx_to_laneid_map.json", "w") as json_file:
        json.dump(lane_id_map, json_file)
    np.save('dataset/HMC/map_files/' + map_abbr + '_' + map_id + "_npyimage_to_city_se2_2019_05_28.npy", city_se2)
    np.save('dataset/HMC/map_files/' + map_abbr + '_' + map_id + "_driveable_area_mat_2019_05_28.npy", drivable_region)
    np.save('dataset/HMC/map_files/' + map_abbr + '_' + map_id + "_ground_height_mat_2019_05_28.npy", ground_height)
    np.save('dataset/HMC/map_files/' + map_abbr + '_' + map_id + "_halluc_bbox_table.npy", halluc_bbox)


def pipe_server():
    # region Config
    pipe_buffer_size = ctypes.sizeof(message_t)
    msg = message_t()
    msg.id = MESSAGE_OK
    message_type = MESSAGE_OK

    # endregion
    pipe = Pipes(IN_PIPE_NAME, OUT_PIPE_NAME, MAX_PIPE_BUFFER_SIZE, connect_first=False)
    print("pipe server")

    raw_data = {'TIMESTAMP': [],
                'TRACK_ID': [],
                'OBJECT_TYPE': [],
                'X': [],
                'Y': [],
                'CITY_NAME': [],
                'HEADING': []}
    mem = DataFrame(raw_data)
    data_count = 0
    while True:
        try:
            status, resp = pipe.read_pipe(pipe_buffer_size)
            if status == 0:
                print("Status:", status)
                if message_type == MESSAGE_OK:
                    # Read message
                    msg = message_t.from_buffer_copy(resp)
                    print("Message: %d" % msg.id)
                    pipe_buffer_size = msg.size
                    message_type = msg.id

                elif message_type == MESSAGE_OBJ:
                    print("OBJ received")
                    num_obj = int(len(resp) / 44)
                    print(num_obj)

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
                        # print(['asdfasdfasdfasdfasdf', cur_time, veh_list[0]])
                        if len(veh_list) > 0 and obj_info['obj_id'] != 0:
                            cur_time = veh_list[0][0]
                            id_mask = '000-0000-0000'
                            obj_id = id_mask[:-len(str(obj_info['obj_id']))] + str(obj_info['obj_id'])
                            ##TODO : 좌표 변환
                            veh_list.append([cur_time, obj_id, 'OTHERS', obj_info['rel_pos_lat'], obj_info['rel_pos_long'], 'HMC', obj_info['heading_angle']])
                        data_temp = pd.DataFrame(veh_list, columns=['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME', 'HEADING'])
                    mem, data_count = update_mem(mem, data_temp, data_count)
                    print(data_count)

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
                elif message_type == MESSAGE_EGO:
                    veh_list = []

                    print("EGO received")

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

                elif message_type == MESSAGE_MAP:
                    # with open("map.data", "wb") as f:
                    #     f.write(resp)

                    print("Map received")
                    num_lane_links = int(len(resp) / 9024)
                    print(len(resp))
                    print(num_lane_links)
                    for i in range(num_lane_links):
                        data = resp[9024 * i:9024 * i + 9024]
                        lane_link = dict()
                        lane_link['next_ids_count'] = int.from_bytes(data[0:4], 'little', signed=False)
                        lane_link['prev_ids_count'] = int.from_bytes(data[88:92], 'little', signed=False)
                        lane_link['next_id'] = []
                        lane_link['prev_id'] = []
                        for num_cnt in range(10):
                            lane_link['next_id'].append(int.from_bytes(data[8 * num_cnt + 8:8 * num_cnt + 16], 'little', signed=False))
                            lane_link['prev_id'].append(int.from_bytes(data[8 * num_cnt + 96:8 * num_cnt + 104], 'little', signed=False))
                        lane_link['ref_lat'] = struct.unpack('<d', data[176:184])[0]
                        lane_link['ref_lng'] = struct.unpack('<d', data[184:192])[0]
                        ref_utm_x, ref_utm_y = myProj(lane_link['ref_lng'], lane_link['ref_lat'])

                        lane_link['points_count'] = int.from_bytes(data[192:196], 'little', signed=False)
                        lane_link['points_x'] = []
                        lane_link['points_y'] = []
                        lane_link['points_x_utm'] = []
                        lane_link['points_y_utm'] = []
                        for num_cnt in range(550):
                            lane_link['points_x'].append(struct.unpack('<d', data[8 * num_cnt + 200:8 * num_cnt + 208])[0])
                            lane_link['points_y'].append(struct.unpack('<d', data[8 * num_cnt + 4600:8 * num_cnt + 4608])[0])
                            lane_link['points_x_utm'].append(struct.unpack('<d', data[8 * num_cnt + 200:8 * num_cnt + 208])[0] + ref_utm_x)
                            lane_link['points_y_utm'].append(struct.unpack('<d', data[8 * num_cnt + 4600:8 * num_cnt + 4608])[0] + ref_utm_y)
                        lane_link['id'] = int.from_bytes(data[9000:9008], 'little', signed=False)
                        lane_link['left_id'] = int.from_bytes(data[9008:9016], 'little', signed=False)
                        lane_link['right_id'] = int.from_bytes(data[9016:9024], 'little', signed=False)

                        file_name = 'saving/map/' + str(lane_link['id']) + '.pkl'
                        if path.exists(file_name) == True:
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
        except:
            print("finished now")
            break
    map_conversion()


pipe_server()
