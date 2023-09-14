import sys
import os
sys.path.append("./thirdparty/RouteGraph")
sys.path.append(os.path.join(os.path.abspath("."),"build"))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import numpy as np
import dgl
import time
import tqdm
import logging
import json

import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace
import PlaceObj
import Timer
from dreamplace.Args import get_args
from dreamplace.CongestionPredictor import CongestionPredictor
PARAM_PATH = 'test/'
DATASET_NAME = "dac2012"
RESULTS_DIR = 'results'

logging.root.name = 'DREAMPlace'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)

netlist_names = [
    # 'superblue1',
    # 'superblue2',
    # 'superblue3',
    # 'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    # 'superblue9',# wrong kreorder
    # 'superblue10',#fail
    'superblue11',
    'superblue12',
    'superblue14',
    # # 'superblue15',#can't read
    'superblue16',
    'superblue18',
    'superblue19',
]

args = get_args()

d = []

for netlist_name in netlist_names:
    # param_path = "/root/DREAMPlace/test/dac2012/superblue19.json"
    param_path = os.path.join(PARAM_PATH, DATASET_NAME, f"{netlist_name}.json")
    params = Params.Params()
    params.load(param_path)

    suffix = ''
    params.__dict__["timing_opt_flag"] = 0
    params.__dict__["our_route_opt"] = 1
    params.__dict__["congestion_weight"] = args.congestion_weight
    params.__dict__["routability_opt_flag"] = 1
    params.__dict__['max_num_area_adjust'] = 4
    params.args = args
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb)
    l_metric = len(metrics)

    path = "%s/%s/%s" % (params.result_dir, args.name, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)
    with open(f"{path}/train.json","w") as f:
        iteration_list = []
        objective_list = []
        overflow_list = []
        wirelength_list = []
        density_list = []
        density_weight_list = []
        max_density_list = []
        gamma_list = []
        for metric in metrics:
            v = metric
            while type(v) == list:
                v = v[0]
            if v.iteration is not None:
                iteration_list.append(int(v.iteration))
            if v.overflow is not None:
                overflow_list.append(float(v.overflow))
            if v.hpwl is not None:
                wirelength_list.append(float(v.hpwl))
            if v.density_weight is not None:
                density_weight_list.append(float(v.density_weight))
            if v.max_density is not None:
                max_density_list.append(float(v.max_density))
            if v.gamma is not None:
                gamma_list.append(float(v.gamma))
        result_list = {
            "iteration":iteration_list,
            "overflow":overflow_list,
            "wirelength":wirelength_list,
            "density_weight":density_weight_list,
            "max_density":max_density_list,
            "gamma":gamma_list,
        }
        json.dump(result_list,f)

