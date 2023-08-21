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
    # 'superblue5',
    # 'superblue6',
    # 'superblue7',
    # 'superblue9',
    # 'superblue10',#fail
    # 'superblue11',
    # 'superblue12',
    # 'superblue14',
    # 'superblue15',
    # 'superblue16',
    # 'superblue18',
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
    params.__dict__["routability_opt_flag"] = 0
    params.args = args
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb)

    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)

