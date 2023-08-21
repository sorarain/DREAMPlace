import os
import sys
import json
import pickle
import time
from functools import reduce
import tqdm

import torch
import numpy as np
import dgl

from thirdparty.RouteGraph.model.RouteGNN import NetlistGNN
from thirdparty.RouteGraph.data.load_data_optimize import torch_feature_grid2node_weighted,node_pairs_among,build_grid_graph

class CongestionPredictor():
    def __init__(self, args, placedb, op_collections, data_collections, params):
        self.initRouteGraphModel(args)

        self.initFixPlaceDBParam(placedb, data_collections)

        self.constructNetlistGraph(placedb, params)

        self.placedb = placedb
        self.params = params
        self.args = args
        self.op_collections = op_collections
        self.data_collections = data_collections

    
    def forward(self, pos):
        pos_ = torch.cat([pos[:self.placedb.num_physical_nodes].unsqueeze(1),pos[self.placedb.num_nodes:self.placedb.num_nodes + self.placedb.num_physical_nodes].unsqueeze(1)],dim=1)
        node_pos_ = pos_.cpu()
        _, hmapfake, vmapfake = self.op_collections.route_utilization_map_op(pos)
        hmapfake, vmapfake = hmapfake.data.cpu(), vmapfake.data.cpu()
        # node_pos[:,0] = node_pos_[:,0].clamp(min=self.placedb.routing_grid_xl,max=self.placedb.routing_grid_xh)
        # node_pos[:,1] = node_pos_[:,1].clamp(min=self.placedb.routing_grid_yl,max=self.placedb.routing_grid_yh)
        node_pos = torch.cat([node_pos_[:,0].clamp(min=self.placedb.routing_grid_xl,max=self.placedb.routing_grid_xh).squeeze().unsqueeze(1),node_pos_[:,1].clamp(min=self.placedb.routing_grid_yl,max=self.placedb.routing_grid_yh).squeeze().unsqueeze(1)],dim=1)

        input_dict = self.constructGraphInput(node_pos, hmapfake, vmapfake)

        
        list_route_graph = self.constructRouteGraph(input_dict, hmapfake, vmapfake)

        congestion = self.subgraphForward(list_route_graph)

        return congestion
    
    def modelForward(self, sub_route_graph):
        in_node_feat = sub_route_graph.nodes['cell'].data['hv']
        in_net_feat = sub_route_graph.nodes['net'].data['hv']
        in_pin_feat = sub_route_graph.edges['pinned'].data['feats']
        in_hanna_feat = sub_route_graph.nodes['gcell'].data['hv']
        pred_cell, _ = self.model.forward(in_node_feat=in_node_feat,in_net_feat=in_net_feat,
                                in_pin_feat=in_pin_feat,in_hanna_feat=in_hanna_feat,node_net_graph=sub_route_graph)
        return pred_cell
    
    def subgraphForward(self, list_route_graph):
        list_cell_congestion = torch.zeros([0], dtype=torch.float32, device=self.device)
        for sub_route_graph in list_route_graph:
            sub_route_graph = sub_route_graph.to(self.device)
            congestion = self.modelForward(sub_route_graph)
            list_cell_congestion = torch.hstack([list_cell_congestion,congestion.squeeze()])
        print("\t\t------------------")
        print(f"\t\t mean congestion {list_cell_congestion.mean()}")
        print(f"\t\t max congestion {list_cell_congestion.max()}")
        print(f"\t\t sum congestion {list_cell_congestion.sum()}")
        print("\t\t------------------")
        return list_cell_congestion.sum()

        
    def constructGraphInput(self, pos, h_net_density_grid, v_net_density_grid):
        input_dict = {}
        input_dict['pos'] = pos
        input_dict['bin_x'], input_dict['bin_y'] = self.bin_x, self.bin_y
        input_dict['hv'] = torch.cat(
                        [
                            torch.tensor(
                                np.stack([self.cell_size_x,
                                        self.cell_size_y,
                                        self.node_pin_num], axis=-1),
                                        dtype=torch.float32
                                        ),
                            torch_feature_grid2node_weighted(
                                    torch.stack([h_net_density_grid, v_net_density_grid], dim=-1),
                                    self.bin_x,
                                    self.bin_y,
                                    pos
                                )
                        ],
            # feature_grid2node(pin_density_grid, 32, 40, node_pos),
            # feature_grid2node(node_density_grid, 32, 40, node_pos),
                        dim=-1
                    )
        
        net_span_feat = []
        ## TODO use cython to rewrite the net_span_feat calculate
        for net,list_pin in tqdm.tqdm(enumerate(self.placedb.net2pin_map),total=len(self.placedb.net2pin_map)):
            xs,ys = [], []
            pxs,pys = [], []
            for pin in list_pin:
                pin = int(pin)
                node = int(self.placedb.pin2node_map[pin])
                x,y = pos[node,:]
                
                pin_px,pin_py = self.placedb.pin_offset_x[pin],self.placedb.pin_offset_y[pin]
                px = x + pin_px
                py = y + pin_py

                xs.append(px)
                ys.append(py)
                pxs.append(int(px / self.bin_x))
                pys.append(int(py / self.bin_y))

            min_x,max_x,min_y,max_y = min(xs),max(xs),min(ys),max(ys)
            span_h = max_x - min_x + 1
            span_v = max_y - min_y + 1
            min_px,max_px,min_py,max_py = min(pxs),max(pxs),min(pys),max(pys)
            span_ph = max_px - min_px + 1
            span_pv = max_py - min_py + 1
            net_span_feat.append([span_h ,span_v, span_h * span_v, 
                                span_ph, span_pv, span_ph * span_pv, len(list_pin)])

        input_dict['net_hv'] = torch.tensor(net_span_feat, dtype=torch.float32)
        # input_dict['net_hv'] = torch.zeros([self.num_nets, 7],dtype=torch.float32)
        return input_dict

    def constructRouteGraph(self, input_dict, h_net_density_grid, v_net_density_grid):
        graphs = []
        for sub_hetero_graph in self.list_hetero_graph:
            sub_hetero_graph.nodes['cell'].data['pos'] = input_dict['pos'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
            sub_hetero_graph.nodes['cell'].data['hv'] = input_dict['hv'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
            sub_hetero_graph.nodes['net'].data['hv'] = input_dict['net_hv'][sub_hetero_graph.nodes['net'].data[dgl.NID]]
            sub_node_pos = input_dict['pos'][sub_hetero_graph.nodes['cell'].data[dgl.NID]].clone().detach()
            sub_route_graph = build_grid_graph(sub_hetero_graph,sub_node_pos,
                    h_net_density_grid,
                    v_net_density_grid,
                    input_dict['bin_x'],input_dict['bin_y'])
            graphs.append(sub_route_graph)
        return graphs

    
    def initFixPlaceDBParam(self, placedb, data_collections):
        self.num_nodes = placedb.num_physical_nodes
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_nets = len(placedb.net_names)
        self.num_bin_x, self.num_bin_y = placedb.num_routing_grids_x,placedb.num_routing_grids_y
        self.bin_x, self.bin_y = placedb.routing_grid_size_x, placedb.routing_grid_size_y

        self.hori_cap = placedb.unit_horizontal_capacity
        self.verti_cap = placedb.unit_vertical_capacity

        horizontal_routing_capacities=torch.from_numpy(
                placedb.unit_horizontal_capacities *
                placedb.routing_grid_size_y)
        vertical_routing_capacities=torch.from_numpy(
            placedb.unit_vertical_capacities *
            placedb.routing_grid_size_x)
        self.nctu_cap = (horizontal_routing_capacities +
                        vertical_routing_capacities).view([
                            1, 1,
                            len(horizontal_routing_capacities)
                        ])
        self.hdm = data_collections.initial_horizontal_utilization_map
        self.vdm = data_collections.initial_vertical_utilization_map

    def initRouteGraphModel(self, args):
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        device = torch.device(args.device)
        self.device = device
        if not args.device == 'cpu':
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(seed)
        
        config = {
        'N_LAYER': args.layers,
        'NODE_FEATS': args.node_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'HANNA_FEATS': args.hanna_feats,
        'EDGE_FEATS': args.edge_feats,
        }

        in_node_feats = 5
        in_net_feats = 7
        in_hanna_feats = 4
        in_pin_feats = 3
        in_edge_feats = 1

        self.model = NetlistGNN(
        in_node_feats=in_node_feats,
        in_net_feats=in_net_feats,
        in_hanna_feats=in_hanna_feats,
        in_pin_feats=in_pin_feats,
        in_edge_feats=in_edge_feats,
        config=config,
        n_target=1,
        # activation=args.outtype,
        topo_conv_type=args.topo_conv_type,
        grid_conv_type=args.grid_conv_type,
        agg_type=args.agg_type,
        cat_raw=args.cat_raw
        ).to(device)
        if args.model:
            model_dicts = torch.load(f'param/{args.model}.pkl', map_location=device)
            self.model.load_state_dict(model_dicts)
            self.model.eval()
        n_param = 0
        for name, param in self.model.named_parameters():
            print(f'\t{name}: {param.shape}')
            n_param += reduce(lambda x, y: x * y, param.shape)
        print(f'# of parameters: {n_param}')
        self.model.train()
    
    def constructNetlistGraph(self, placedb, params):
        us,vs = placedb.pin2node_map,placedb.pin2net_map
        cell_size_x = placedb.node_size_x
        cell_size_y = placedb.node_size_y
        node_pin_num = np.zeros(self.num_nodes,dtype=np.float32)
        net_degree = np.zeros(self.num_nets,dtype=np.float32)
        pin_feats = []

        for node,list_pin in enumerate(placedb.node2pin_map):
            node_pin_num[node] = len(list_pin)
            for pin in list_pin:
                net = int(placedb.pin2net_map[pin])
                net_degree[net] += 1
                if placedb.pin_direct[pin] == b'OUTPUT':
                    pin_IO = 0
                else:
                    pin_IO = 1
                pin_feats.append([placedb.pin_offset_x[pin],placedb.pin_offset_y[pin],pin_IO])

        us_homo,vs_homo = [],[]
        for net,list_pin in tqdm.tqdm(enumerate(placedb.net2pin_map),total=len(placedb.net2pin_map)):
            nodes = []
            for pin in list_pin:
                pin = int(pin)
                node = int(placedb.pin2node_map[pin])
                nodes.append(node)
            us_, vs_ = node_pairs_among(np.array(nodes,dtype=np.int32), max_cap=8)
            us_homo.extend(us_)
            vs_homo.extend(vs_)
        self.homo_graph = dgl.add_self_loop(dgl.graph((us_homo, vs_homo), num_nodes=self.num_nodes))
        p_gs = dgl.metis_partition(self.homo_graph,int(np.ceil(self.num_nodes/10000)))
        self.partition_list = []
        for k,val in p_gs.items():
            nids = val.ndata[dgl.NID].numpy().tolist()
            self.partition_list.append(nids)

        
        self.hetero_graph = dgl.heterograph({
            ('cell','pins','net'):(us,vs),
            ('net','pinned','cell'):(vs,us),
        },num_nodes_dict={'cell':self.num_nodes,'net':self.num_nets})
        # self.node_hv = torch.zeros([self.num_nodes,7],dtype=torch.float32)
        # self.net_hv = torch.zeros([self.num_nets,7],dtype=torch.float32)
        self.pin_feats = torch.tensor(pin_feats,dtype=torch.float32)

        # self.node_hv[:,0] = torch.tensor(cell_size_x[:self.num_nodes],dtype=torch.float32)
        # self.node_hv[:,1] = torch.tensor(cell_size_y[:self.num_nodes],dtype=torch.float32)
        # self.node_hv[:,2] = torch.tensor(node_pin_num,dtype=torch.float32)
        self.node_pin_num = torch.tensor(node_pin_num,dtype=torch.float32)
        self.cell_size_x = torch.tensor(cell_size_x[:self.num_nodes],dtype=torch.float32)
        self.cell_size_y = torch.tensor(cell_size_y[:self.num_nodes],dtype=torch.float32)
        
        # self.net_hv[:,-1] = torch.tensor(net_degree,dtype=torch.float32)
        self.net_degree = torch.unsqueeze(torch.tensor(net_degree,dtype=torch.float32),dim=-1)

        self.hetero_graph.edges['pinned'].data['feats'] = self.pin_feats
        self.hetero_graph.nodes['net'].data['degree'] = self.net_degree

        self.list_hetero_graph = []
        all_net_degree_dict = {}
        node_belong_partition = np.ones(self.hetero_graph.num_nodes(ntype='cell'))
        for i,partition in enumerate(self.partition_list):
            for node in list(set(partition)):
                node_belong_partition[node] = i
        for net_id, node_id in zip(*[ns.tolist() for ns in self.hetero_graph.edges(etype='pinned')]):
            belong = node_belong_partition[node_id]
            all_net_degree_dict.setdefault(belong,{}).setdefault(net_id,0)
            all_net_degree_dict[belong][net_id] += 1
        for i,partition in enumerate(self.partition_list):
            partition_set = set(partition)
            new_net_degree_dict = all_net_degree_dict[i]
            keep_nets_id = np.array(list(new_net_degree_dict.keys()))
            keep_nets_degree = np.array(list(new_net_degree_dict.values()))
            part_hetero_graph = dgl.node_subgraph(self.hetero_graph, nodes={'cell': partition, 'net': keep_nets_id})
            self.list_hetero_graph.append(part_hetero_graph)






