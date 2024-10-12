import networkx as nx
import numpy as np
import random
import copy

import geopandas as gpd
import pandas as pd
from shapely import geometry
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import plotly.express as px

# transit line
class TLine(object):
    tot_seq = 0

    def __init__(self, name, stations, start, end, vstart=None, vend=None, vline=None) -> None:
        TLine.tot_seq += 1
        self.name = name
        self.seq = TLine.tot_seq
        self.stations = stations

        self.start_id = start[0]
        self.start = np.array(start[1])
        self.end_id = end[0]
        self.end = np.array(end[1])
        if vstart is None:
            self.vstart = self.start - self.end
        else:
            self.vstart = vstart
        if vend is None:
            self.vend = self.end - self.start
        else:
            self.vend = vend
        if vline is None:
            self.vline = self.start - self.end
        else:
            self.vline = vline

    def add_station(self, new_station, is_start):
        if is_start:
            tmp_pos = np.array(new_station[1])
            self.vstart = tmp_pos - self.start
            self.start = tmp_pos
            self.start_id = new_station[0]
            self.stations.insert(0, new_station[0])
        else:
            tmp_pos = np.array(new_station[1])
            self.vend = tmp_pos - self.end
            self.end = tmp_pos
            self.end_id = new_station[0]
            self.stations.append(new_station[0])

    def line_expand(self, old_station, new_station):
        if old_station[0] == self.start_id:
            self.add_station(new_station, True)
        elif old_station[0] == self.end_id:
            self.add_station(new_station, False)
        else:
            raise RuntimeError('[!]Expand failed')
        
    def get_start_and_end_pos(self):
        return self.start, self.end
    
    def get_start_and_end_id(self):
        return self.start_id, self.end_id
    
    def get_start_and_end_v(self):
        return self.vstart, self.vend
    
    def get_stations_num(self):
        return len(self.stations)
    
    def get_stations(self):
        return self.stations
    


class Transit(object):
    DUMMY_STATION = 0

    def __init__(self, data, budget, max_line, min_num, expand, cost, origin=None, destination=None, shape_param=None) -> None:

        self.G, self.region_processed, self.od_pair, self.line_info = data
        self.G = nx.Graph(self.G)

        self.cost_per_kmeter,self.cost_per_station,self.cost_per_trans_station = cost
        self.budget = budget
        self.max_line = max_line
        self.min_num = min_num
        self.build_line_num = 0
        self.expand = expand

        self.total_cost = 0
        self.total_od = 0

        self.node_list = [n for n in self.G.nodes()]
        self.node_list_idx = dict(zip(self.node_list, [i for i in range(len(self.node_list))]))
        self.node_num = len(self.node_list)
        self.node_pos = nx.get_node_attributes(self.G, 'pos')

        self.edge_list = [e for e in self.G.edges()]
        self.edge_build_idx = []
        attributes_idx = {}
        for idx in range(len(self.edge_list)):
            attributes_idx[self.edge_list[idx]] = {}
            attributes_idx[self.edge_list[idx]]['idx'] = idx
            attributes_idx[self.edge_list[idx]]['metro_dis'] = 1e6
        nx.set_edge_attributes(self.G, attributes_idx)

        self.done = 0
        self.build_new_line = True
        self.building_line = None

        self.build_node = []
        self.build_node_od = {}
        self.station = {node: 0 for node in self.node_list}
        self.seq = {node: [0,0,0,0] for node in self.node_list}
        self.build_edge = []
        self.metro_lines = []
        self.min_dis = {}

        self.env_init()

    def env_init(self):
        self.init_build_line()
        self.init_min_dis()
        self.init_mask()
        self.init_feature()

    def init_build_line(self):
        self.metro_lines = []

        line_seq, lines = copy.deepcopy(self.line_info)
        self.init_lines = line_seq

        for line_name in self.init_lines:
            start_id = lines[line_name][0]
            start_pos = self.node_pos[start_id]
            vstart = self.node_pos[start_id] - self.node_pos[lines[line_name][1]]
            end_id = lines[line_name][-1]
            end_pos = self.node_pos[end_id]
            vend = self.node_pos[end_id] - self.node_pos[lines[line_name][-2]]
            vline = start_pos - end_pos
            tmp_line = TLine(line_name, lines[line_name], [start_id,start_pos], [end_id,end_pos], vstart, vend, vline)
            self.metro_lines.append(tmp_line)

            for line_idx in range(len(lines[line_name])-1):
                pre = lines[line_name][line_idx]
                cur = lines[line_name][line_idx + 1]
                if not self.station[pre]:
                    self.build_node.append(pre)
                if not self.station[cur]:
                    self.build_node.append(cur)
                if pre != cur:
                    self.station[pre] += 1
                    self.station[cur] += 1


                    if line_idx == 0:
                        self.seq[pre][:2] = [tmp_line.seq, line_idx + 1]

                    if self.seq[cur][:2] == [0,0]:
                        self.seq[cur][:2] = [tmp_line.seq, line_idx + 2]
                    else:
                        self.seq[cur][2:] = [tmp_line.seq, line_idx + 2]

                    self.G[pre][cur]['metro_dis'] = self.G[pre][cur]['length']
                    idx = self.G[pre][cur]['idx']
                    self.edge_build_idx.append(idx)

        self.pre_build_node = copy.deepcopy(self.build_node)
        for node in self.node_list:
            self.build_node_od[node] = 0
            for build_node in self.build_node:
                self.build_node_od[node] += (self.od_pair[node][build_node] + self.od_pair[build_node][node])

    def init_min_dis(self):
        metro_min_dis = dict(nx.shortest_path_length(self.G,weight='metro_dis'))
        self.min_dis_copy = copy.deepcopy(metro_min_dis)
        for n1 in self.build_node:
            self.min_dis[n1] = {}
            for n2 in self.build_node:
                self.min_dis[n1][n2] = metro_min_dis[n1][n2]

        for e in self.G.edges():
            self.G.add_edge(e[0],e[1],length=self.G[e[0]][e[1]]['length'],\
                            cost=self.G[e[0]][e[1]]['length'] * self.cost_per_kmeter + self.cost_per_station)

    def init_mask(self):
        self._cal_mask()

    def init_feature(self):
        self._cal_node_centrality()
        self._cal_node_od()
        self._cal_node_degree()
        self._cal_graph_node_feature()
        self._cal_edge_index()

    def reset(self,eval=False):
        self.build_new_line = True
        self.building_line = None
        self.done = 0
        self.build_line_num = 0

        self.build_node = []
        self.build_edge = []
        self.edge_build_idx = []
        self.station = {node: 0 for node in self.node_list}
        self.seq = {node: [0,0,0,0] for node in self.node_list}
        TLine.tot_seq = 0
        self.build_node_od = {}
        self.init_build_line()
        self.min_dis = copy.deepcopy(self.min_dis_copy)

        self.reward = 0
        self.total_cost = 0
        self.total_od = 0
        self.init_mask()
        self._cal_node_endpoint()

    def _get_node_loc(self,n):
        # beijing
        longtitude_min = 116.161 - 0.01
        longtitude_max = 116.753 + 0.01
        latitude_min = 39.670 - 0.01
        latitude_max = 40.130 + 0.01

        x = (self.node_pos[n][0] - latitude_min) / (latitude_max - latitude_min)
        y = (self.node_pos[n][1] - longtitude_min) / (longtitude_max - longtitude_min)
        return [x, y]
    
    def _get_node_pre_build(self,n):
        if n in self.pre_build_stations:
            return 1
        else: 
            return 0
        
    def _cal_node_centrality(self):
        degree_cen = nx.degree_centrality(self.G)
        betweenness_cen = nx.betweenness_centrality(self.G,weight='length')
        eigenvector_cen = nx.eigenvector_centrality_numpy(self.G,weight='length')
        closeness_cen = nx.closeness_centrality(self.G, distance='length')
        self.node_centrality = {}
        for node in self.node_list:
            self.node_centrality[node] = [degree_cen[node], betweenness_cen[node],eigenvector_cen[node], closeness_cen[node]]

    def _cal_node_od(self):
        self.node_od = {}
        for node in self.node_list:
            self.node_od[node] = [self.region_processed[node]['in']/1e5 , self.region_processed[node]['out']/1e5]
            near_node_od = 0
            for near in list(nx.neighbors(self.G,node)):
                try:
                    near_node_od += (self.od_pair[node][near] + self.od_pair[near][node])
                except:
                    raise RuntimeError(node,near)

            self.node_od[node].append(near_node_od/1e5)

    def _cal_graph_node_feature(self):
        self.graph_node_feature = {}
        for node in self.node_list:
            self.graph_node_feature[node] = self._get_node_loc(node) + self.node_centrality[node] + [self.node_degree_total[node]/1e1]\
                                            + [self.station[node]] + self.region_processed[node]['feature'] + self.node_od[node]
            self.graph_node_feature[node][7] = self.graph_node_feature[node][7]/1e1
            self.graph_node_feature[node][12] = self.graph_node_feature[node][12]/1e1
            self.graph_node_feature[node][17:] = [0] * len(self.graph_node_feature[node][17:])

        
        
        self.node_centrality = None
        self.node_od = None
        self.region_processed = None
        self.node_degree_total = None

    def _cal_node_endpoint(self):
        is_endpoint = [[0,0,0]] * self.node_num
        self.node_is_endpoint = dict(zip(self.node_list, is_endpoint))

        for line in self.metro_lines:
            s,e = line.get_start_and_end_id()
            self.node_is_endpoint[s][0] = 1
            self.node_is_endpoint[e][0] = 1

        for node in self.pre_build_node:
                self.node_is_endpoint[node][1] = 1
       
        for node in self.build_node:
                self.node_is_endpoint[node][2] = 1

    def _cal_graph_node_feature_dim(self):
        return 26
    
    def get_numerical_dim(self):
        return 7
    
    def get_node_dim(self):
        return self._cal_graph_node_feature_dim() + 5 + 4

    def _cal_edge_index(self):
        self.edge_index_dis = []
        self.edge_index_od = []

        for e in self.edge_list:
            idx1 = self.node_list_idx[e[0]]
            idx2 = self.node_list_idx[e[1]]
            self.edge_index_dis.append([idx1, idx2])

        for n1 in self.node_list:
            for n2 in self.od_pair[n1]:
                if n2 in self.node_list:
                    if self.od_pair[n1][n2] > 1e3 or self.od_pair[n2][n1] > 1e3:
                        idx1 = self.node_list_idx[n1]
                        idx2 = self.node_list_idx[n2]
                        if idx1 > idx2:
                            self.edge_index_od.append([idx1, idx2])
        

    def _cal_node_degree(self):
        self.node_degree_total = {}
        for n in self.node_list:
            self.node_degree_total[n] = len(list(self.G.neighbors(n)))
    
    def _get_node_feature(self,node):
        return self.graph_node_feature[node] + [self.build_node_od[node]/5e5] + self.node_is_endpoint[node] + [self.station[node]] + self.seq[node]

    def _get_numerical(self):

        numerical = [len(self.metro_lines),self.total_cost/self.budget,self.cost_per_kmeter/self.budget,self.cost_per_station/self.budget,self.cost_per_trans_station/self.budget]
        if self.build_new_line:
            numerical = numerical + [1,0]
        else:
            numerical = numerical + [0,1] 
                            
        return numerical

    def get_obs(self):
        numerical = self._get_numerical()
        node_feature = np.concatenate([[self._get_node_feature(n) for n in self.node_list]], axis=1)
        
        mask = self.get_mask()
        return numerical, node_feature, self.edge_index_dis, self.edge_index_od, mask
    
    def update_dis_cost_reward(self,old_station,new_station,is_transfer,is_new_line):
        self.reward = 0
        cost = 0
            
        if not is_transfer:
            self.min_dis[new_station] = {}
            for n in self.build_node:
                # try:
                self.min_dis[new_station][new_station] = 0
                self.min_dis[new_station][n] = self.min_dis[n][old_station] + self.G[new_station][old_station]['length']
                self.min_dis[n][new_station] = self.min_dis[new_station][n]

                linedis = haversine(self.node_pos[n], self.node_pos[new_station], unit=Unit.KILOMETERS)

                if linedis <= 3:
                    self.reward += (self.od_pair[n][new_station] + self.od_pair[new_station][n]) / 1e5
                else:
                    metrodis = self.min_dis[n][new_station]
                    self.reward += (self.od_pair[n][new_station] + self.od_pair[new_station][n]) / 1e5 * min(max(0.2, linedis/metrodis), 0.8)

            self.build_node.append(new_station)
            cost += self.G[new_station][old_station]['cost']

        else:
            for n in self.build_node:
                linedis1 = haversine(self.node_pos[n], self.node_pos[new_station], unit=Unit.KILOMETERS)
                linedis2 = haversine(self.node_pos[n], self.node_pos[old_station], unit=Unit.KILOMETERS)

                min_dis1 = min(self.min_dis[n][new_station], self.min_dis[n][old_station] + self.G[old_station][new_station]['length'])
                min_dis2 = min(self.min_dis[n][old_station], self.min_dis[n][new_station] + self.G[new_station][old_station]['length'])

                if linedis1 > 3:
                    metrodis = self.min_dis[n][new_station]
                    r1 = min(max(0.2, linedis1/metrodis), 0.8)
                    r2 = min(max(0.2, linedis1/min_dis1), 0.8)
                    if min_dis1 < metrodis:
                        self.reward += (self.od_pair[n][new_station] + self.od_pair[new_station][n]) / 1e5 * (r2 - r1)

                if n != new_station:
                    if linedis2 > 3:
                        metrodis = self.min_dis[n][old_station]
                        r1 = min(max(0.2, linedis2/metrodis), 0.8)
                        r2 = min(max(0.2, linedis2/min_dis2), 0.8)
                        if min_dis2 < metrodis:
                            self.reward += (self.od_pair[n][old_station] + self.od_pair[old_station][n]) / 1e5 * (r2 - r1)

                self.min_dis[n][new_station] = min_dis1
                self.min_dis[new_station][n] = min_dis1
                self.min_dis[n][old_station] = min_dis2
                self.min_dis[old_station][n] = min_dis2

            transfer_count = 0
            if self.station[new_station] <= 2:
                transfer_count += 1
            if self.station[old_station] <= 2:
                transfer_count += 1

            cost += self.G[new_station][old_station]['cost'] - self.cost_per_station + transfer_count * self.cost_per_trans_station

        self.total_od += self.reward
        self.total_cost += cost
        self.station[new_station] += 1
        self.station[old_station] += 1
        self.edge_build_idx.append(self.G[new_station][old_station]['idx'])


    def add_station_from_action(self,action):
        new_station = self.node_list[action]
        is_transfer = False
        is_new_line = False
        if new_station in self.build_node:
            is_transfer = True
        else:
            is_transfer = False

        if new_station in self.build_old_line_dict:
            is_new_line = False
            old_station = self.build_old_line_dict[new_station][0]
            cur_line = self.build_old_line_dict[new_station][1]
            cur_line.line_expand([old_station, self.node_pos[old_station]],  [new_station, self.node_pos[new_station]])
            self.node_is_endpoint[old_station][0] = 0
            self.node_is_endpoint[new_station][0] = 1
            self.node_is_endpoint[new_station][2] = 1

            if cur_line.get_stations_num() >= self.min_num and self.build_line_num < self.max_line:
                self.build_new_line = True
            else:
                self.build_new_line = False
            self.building_line = cur_line

        elif new_station in self.build_new_line_dict:
            is_new_line = True
            old_station = self.build_new_line_dict[new_station]
            new_metro_line = TLine('new_{}'.format(len(self.metro_lines)+1), [old_station,new_station],\
                                   [old_station, self.node_pos[old_station]], [new_station, self.node_pos[new_station]])

            self.build_line_num += 1
            
            self.node_is_endpoint[old_station][0] = 1
            self.node_is_endpoint[new_station][0] = 1
            self.node_is_endpoint[new_station][0] = 1

            if self.seq[old_station][:2] == [0,0]:
                self.seq[old_station][:2] = [new_metro_line.seq, 1]
            else:
                self.seq[old_station][2:] = [new_metro_line.seq, 1]


            self.metro_lines.append(new_metro_line)
            self.build_new_line = False
            
            self.building_line = new_metro_line

        if self.seq[new_station][:2] == [0,0]:
            self.seq[new_station][:2] = [self.building_line.seq, self.building_line.get_stations_num()]
        else:
            self.seq[new_station][2:] = [self.building_line.seq, self.building_line.get_stations_num()]

        if old_station not in self.build_node:
            raise RuntimeError('[!]Invalid old station: ', old_station)
        self.build_edge.append([old_station,new_station])
        self.update_dis_cost_reward(old_station,new_station,is_transfer,is_new_line)

        if not is_transfer:
            for node in self.node_list:
                    self.build_node_od[node] += (self.od_pair[node][new_station] + self.od_pair[new_station][node])

    def _cal_mask(self):
        mask = [0] * self.node_num

        self.build_old_line_dict = {}
        stations = []
        if not False:
            if self.build_line_num >= self.max_line and self.building_line.get_stations_num() >= self.min_num:
                lines = self.metro_lines
            elif not self.build_new_line or not self.expand:
                lines = [self.building_line]
                if lines[0] is None:
                    lines = self.metro_lines
            else:
                lines = self.metro_lines
            
            for line in lines:
                    stations += line.get_stations()

            for line in lines:
                idstart, idend = line.get_start_and_end_id()
                vstart, vend = line.get_start_and_end_v()
                
                ns = nx.neighbors(self.G, idstart)
                for n in ns:
                    if self.G[idstart][n]['idx'] not in self.edge_build_idx:
                        v1 = vstart
                        v2 = np.array(self.node_pos[n]) - np.array(self.node_pos[idstart])
                        if np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6) > 0.3 and np.dot(line.vline, v2)/(np.linalg.norm(line.vline)*np.linalg.norm(v2) + 1e-6) > 0.3 \
                            and self.total_cost + self.G[n][idstart]['cost'] <= self.budget:

                            self.build_old_line_dict[n] = [idstart,line]
                            idx = self.node_list_idx[n]
                            mask[idx] = 1

                ne = nx.neighbors(self.G, idend)
                for n in ne:
                    if self.G[idend][n]['idx'] not in self.edge_build_idx:
                        v1 = vend
                        v2 = np.array(self.node_pos[n]) - np.array(self.node_pos[idend])
                        if np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6) > 0.3 and np.dot(line.vline, v2)/(np.linalg.norm(line.vline)*np.linalg.norm(v2) + 1e-6) < -0.3 \
                        and self.total_cost + self.G[n][idend]['cost'] <= self.budget:
                            self.build_old_line_dict[n] = [idend,line]
                            idx = self.node_list_idx[n]
                            mask[idx] = 1

        self.build_new_line_dict = {}
        if self.build_new_line:
            for bn in self.pre_build_node:
                ns = nx.neighbors(self.G, bn)
                for n in ns:
                    if self.G[bn][n]['idx'] not in self.edge_build_idx and n not in self.build_old_line_dict and self.total_cost + self.G[n][bn]['cost'] <= self.budget:
                        if n not in self.build_new_line_dict:
                            self.build_new_line_dict[n] = bn
                        else:
                            if self.G[bn][n]['length'] < self.G[self.build_new_line_dict[n]][n]['length']:
                                self.build_new_line_dict[n] = bn
                        idx = self.node_list_idx[n]
                        mask[idx] = 1

            for n in self.build_node:
                if n not in self.pre_build_node:
                    ns = nx.neighbors(self.G, n)
                    for nn in ns:
                        idx = self.node_list_idx[nn]
                        mask[idx] = 0

        policy = 0
        if policy == 1:
            mask = [0] * self.node_num
            if self.build_new_line:
                for n in self.build_new_line_dict:
                    idx = self.node_list_idx[n]
                    mask[idx] = 2
            else:
                for n in self.build_old_line_dict:
                    idx = self.node_list_idx[n]
                    mask[idx] = 2

        elif policy == 2:
            mask = [0] * self.node_num
            od_max = 0
            id_max = 0

            if self.build_new_line:
                for n in self.build_new_line_dict:
                    idx = self.node_list_idxget_maskself.mask[n]
                    mask[idx] = 2
            else:
                for n in self.build_old_line_dict:
                    idx = self.node_list_idx[n]
                    if self.build_node_od[n] >= od_max:
                        od_max = self.build_node_od[n]
                        id_max = idx
                if id_max != 0:
                    mask[id_max] = 2


        if np.array(mask).sum() == 0:
            self.done = 1
        else:
            self.done = 0

        self.mask = mask

    def get_mask(self):
        return self.mask

    def fake_cost(self,action):
        new_station = self.node_list[action]

        if new_station in self.build_old_line_dict:
            old_station = self.build_old_line_dict[new_station][0]
        elif new_station in self.build_new_line_dict:
            old_station = self.build_new_line_dict[new_station]
        else:
            raise RuntimeError('[!]Invalid action: ',new_station)

        transfer_count = 0
        if self.station[new_station] <= 2:
            transfer_count += 1
        if self.station[old_station] <= 2:
            transfer_count += 1

        return self.G[new_station][old_station]['cost'] - self.cost_per_station + transfer_count * self.cost_per_trans_station

    def get_reward(self):
        if self.done and self.budget > 3:
            return self.reward + self.total_cost - self.budget
        else:
            return self.reward
    
    def get_cost(self):
        return self.total_cost
    
    def get_od(self):
        return self.total_od
    
    def get_done(self):
        self._cal_mask()
        return self.done



