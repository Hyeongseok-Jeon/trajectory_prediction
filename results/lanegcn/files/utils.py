# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import numpy as np
import sys
import cv2
import os

import torch
from torch import optim
import copy
from scipy import sparse


def read_argo_data(city, df):

    """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
    agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
    mapping = dict()
    for i, ts in enumerate(agt_ts):
        mapping[ts] = i

    trajs = np.concatenate((
        df.X.to_numpy().reshape(-1, 1),
        df.Y.to_numpy().reshape(-1, 1)), 1)

    steps = [mapping[x] for x in df['TIMESTAMP'].values]
    steps = np.asarray(steps, np.int64)

    objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
    keys = list(objs.keys())
    obj_type = [x[1] for x in keys]

    agt_idx = obj_type.index('AGENT')
    idcs = objs[keys[agt_idx]]

    agt_traj = trajs[idcs]
    agt_step = steps[idcs]

    del keys[agt_idx]
    ctx_trajs, ctx_steps = [], []
    for key in keys:
        idcs = objs[key]
        ctx_trajs.append(trajs[idcs])
        ctx_steps.append(steps[idcs])

    data = dict()
    data['city'] = city
    data['trajs'] = [agt_traj] + ctx_trajs
    data['steps'] = [agt_step] + ctx_steps
    return data


def get_obj_feats(data):
    orig = data['trajs'][0][19].copy().astype(np.float32)

    pre = data['trajs'][0][18] - orig
    theta = np.pi - np.arctan2(pre[1], pre[0])

    rot = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], np.float32)

    feats, ctrs, gt_preds, has_preds = [], [], [], []
    for traj, step in zip(data['trajs'], data['steps']):
        if 19 not in step:
            continue

        gt_pred = np.zeros((30, 2), np.float32)
        has_pred = np.zeros(30, bool)
        future_mask = np.logical_and(step >= 20, step < 50)
        post_step = step[future_mask] - 20
        post_traj = traj[future_mask]
        gt_pred[post_step] = post_traj
        has_pred[post_step] = 1

        obs_mask = step < 20
        step = step[obs_mask]
        traj = traj[obs_mask]
        idcs = step.argsort()
        step = step[idcs]
        traj = traj[idcs]

        for i in range(len(step)):
            if step[i] == 19 - (len(step) - 1) + i:
                break
        step = step[i:]
        traj = traj[i:]

        feat = np.zeros((20, 3), np.float32)
        feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
        feat[step, 2] = 1.0

        x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
        if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
            continue

        ctrs.append(feat[-1, :2].copy())
        feat[1:, :2] -= feat[:-1, :2]
        feat[step[0], :2] = 0
        feats.append(feat)
        gt_preds.append(gt_pred)
        has_preds.append(has_pred)

    feats = np.asarray(feats, np.float32)
    ctrs = np.asarray(ctrs, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    has_preds = np.asarray(has_preds, bool)

    data['feats'] = feats
    data['ctrs'] = ctrs
    data['orig'] = orig
    data['theta'] = theta
    data['rot'] = rot
    data['gt_preds'] = gt_preds
    data['has_preds'] = has_preds
    return data


def get_lane_graph(data, am):
    """Get a rectangle area defined by pred_range."""
    x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
    radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
    lane_ids = am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
    lane_ids = copy.deepcopy(lane_ids)

    lanes = dict()
    for lane_id in lane_ids:
        lane = am.city_lane_centerlines_dict[data['city']][lane_id]
        lane = copy.deepcopy(lane)
        centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
        x, y = centerline[:, 0], centerline[:, 1]
        if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
            continue
        else:
            """Getting polygons requires original centerline"""
            polygon = am.get_lane_segment_polygon(lane_id, data['city'])
            polygon = copy.deepcopy(polygon)
            lane.centerline = centerline
            lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
            lanes[lane_id] = lane

    lane_ids = list(lanes.keys())
    ctrs, feats, turn, control, intersect = [], [], [], [], []
    for lane_id in lane_ids:
        lane = lanes[lane_id]
        ctrln = lane.centerline
        num_segs = len(ctrln) - 1

        ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
        feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

        x = np.zeros((num_segs, 2), np.float32)
        if lane.turn_direction == 'LEFT':
            x[:, 0] = 1
        elif lane.turn_direction == 'RIGHT':
            x[:, 1] = 1
        else:
            pass
        turn.append(x)

        control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
        intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    num_nodes = count

    pre, suc = dict(), dict()
    for key in ['u', 'v']:
        pre[key], suc[key] = [], []
    for i, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]
        idcs = node_idcs[i]

        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]
        if lane.predecessors is not None:
            for nbr_id in lane.predecessors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre['u'].append(idcs[0])
                    pre['v'].append(node_idcs[j][-1])

        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]
        if lane.successors is not None:
            for nbr_id in lane.successors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc['u'].append(idcs[-1])
                    suc['v'].append(node_idcs[j][0])

    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]

        nbr_ids = lane.predecessors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_pairs.append([i, j])

        nbr_ids = lane.successors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_pairs.append([i, j])

        nbr_id = lane.l_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.r_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])
    pre_pairs = np.asarray(pre_pairs, np.int64)
    suc_pairs = np.asarray(suc_pairs, np.int64)
    left_pairs = np.asarray(left_pairs, np.int64)
    right_pairs = np.asarray(right_pairs, np.int64)

    graph = dict()
    graph['ctrs'] = np.concatenate(ctrs, 0)
    graph['num_nodes'] = num_nodes
    graph['feats'] = np.concatenate(feats, 0)
    graph['turn'] = np.concatenate(turn, 0)
    graph['control'] = np.concatenate(control, 0)
    graph['intersect'] = np.concatenate(intersect, 0)
    graph['pre'] = [pre]
    graph['suc'] = [suc]
    graph['lane_idcs'] = lane_idcs
    graph['pre_pairs'] = pre_pairs
    graph['suc_pairs'] = suc_pairs
    graph['left_pairs'] = left_pairs
    graph['right_pairs'] = right_pairs

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

    for key in ['pre', 'suc']:
        graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], 6)
    return graph


def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs

def index_dict(data, idcs):
    returns = dict()
    for key in data:
        returns[key] = data[key][idcs]
    return returns


def rotate(xy, theta):
    st, ct = torch.sin(theta), torch.cos(theta)
    rot_mat = xy.new().resize_(len(xy), 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct
    xy = torch.matmul(rot_mat, xy.unsqueeze(2)).view(len(xy), 2)
    return xy


def merge_dict(ds, dt):
    for key in ds:
        dt[key] = ds[key]
    return


class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

class Optimizer(object):
    def __init__(self, params, config, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        opt = config["opt"]
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"], weight_decay=config["wd"]
            )
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = config["lr_func"]

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]
