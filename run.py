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
import torch
from data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn
from torch.utils.data import DataLoader
from utils import Logger, load_pretrain, gpu


from xml.etree.ElementTree import Element, SubElement, ElementTree, dump
import json
pd.set_option('display.max_rows', None)

MAX_PIPE_BUFFER_SIZE = 1024 * 1024 * 1024
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
config["preprocess"] = False  # we use raw data to generate preprocess data
config["val_workers"] = 1
config["workers"] = 1
config['cross_dist'] = 6
config['cross_angle'] = 0.5 * np.pi
config["batch_size"] = 1
config["val_batch_size"] = 1
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


def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data


def preprocess(graph, cross_dist, cross_angle=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def modify(config, data_tot, save):
    t = time.time()

    for i, datus in enumerate(data_tot):
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            graph[key] = torch.from_numpy(ref_copy(datus['graph'][key]))
        graph['idx'] = i
        data = [graph]
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))

        for j, graph in enumerate(out):
            idx = graph['idx']
            data_tot[idx]['graph']['left'] = graph['left']
            data_tot[idx]['graph']['right'] = graph['right']

    return data_tot

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

        data_tot = [None for x in range(len(full_id))]

        for i in range(len(full_id)):
            data_save = mem.copy()

            indices = [j for j, x in enumerate(list(data_save['TRACK_ID'])) if x == full_id[i]]
            for k in indices:
                data_save.iat[k, 2] = 'AGENT'
            data_input = read_argo_data('HMC',data_save)
            data_input = get_obj_feats(data_input)
            data_input['graph'] = get_lane_graph(data_input, am)
            data_input['idx'] = i

            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data_input[key])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            data_tot[store["idx"]] = store
            if i > 10:
                break
        print(['a0', time.time()-init])

        data = modify(config, data_tot, config["preprocess_train"])
        print(['a', time.time()-init])
        data_input = dict()
        for key in [
            "city",
            "orig",
            "gt_preds",
            "has_preds",
            "theta",
            "rot",
            "feats",
            "ctrs",
            "graph",
        ]:
            data_input[key] = []

        for i in range(len(data)):
            for key in [
                "city",
                "orig",
                "gt_preds",
                "has_preds",
                "theta",
                "rot",
                "feats",
                "ctrs",
                "graph",
            ]:
                if key in ['orig', 'gt_preds', 'has_preds','rot', 'feats', 'ctrs']:
                    data_input[key].append(torch.from_numpy(data[i][key]))
                elif key in ['city', 'theta']:
                    data_input[key].append(data[i][key])
                elif key in ['graph']:
                    graph_temp = dict()
                    for keys in [
                        'ctrs',
                        'num_nodes',
                        'feats',
                        'turn',
                        'control',
                        'intersect',
                        'pre',
                        'suc',
                        'lane_idcs',
                        'pre_pairs',
                        'suc_pairs',
                        'left_pairs',
                        'right_pairs',
                        'left',
                        'right'
                    ]:
                        if keys in ['num_nodes']:
                            graph_temp[keys] = data[i][key][keys]
                        elif keys in ['pre','suc']:
                            list_temp = []
                            for j in range(6):
                                dict_tmp = dict()
                                dict_tmp['u'] = torch.from_numpy(data[i][key][keys][j]['u'])
                                dict_tmp['v'] = torch.from_numpy(data[i][key][keys][j]['v'])
                                list_temp.append(dict_tmp)
                            graph_temp[keys] = list_temp
                        elif keys in ['left','right']:
                            dict_temp = dict()
                            dict_temp['u'] = torch.from_numpy(data[i][key][keys]['u'])
                            dict_temp['v'] = torch.from_numpy(data[i][key][keys]['v'])
                            graph_temp[keys] = dict_temp
                        else:
                            graph_temp[keys] = torch.from_numpy(data[i][key][keys])
                    data_input[key].append(graph_temp)
        print(['b', time.time()-init])

        output = net(data_input)

        print('conversion finished', time.time()-init)

        new_mem = mem[mem['TIMESTAMP'] != mem['TIMESTAMP'][0]]
        new_mem = new_mem.reset_index(drop=True)
    else:
        new_mem = pd.concat([mem, new], ignore_index=True)


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
    # win32pipe.WaitNamedPipe(r'{}'.format(IN_PIPE_NAME), win32pipe.NMPWAIT_WAIT_FOREVER)

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
                if data_num > 1500:
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
