from pvdiffusion import PVDiffusion
import os
from configs.infer_config import get_parser
from decord import VideoReader, cpu
import torch
from utils.pvd_utils import *

if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir,exist_ok=True)
    pvd = PVDiffusion(opts)

    if opts.mode == 'single_view_specify':
        pvd.nvs_single_view()

    elif opts.mode == 'single_view_ref_iterative':
        all_results = pvd.nvs_single_view_ref_iterative()

    elif opts.mode == 'single_view_1drc_iterative':
        all_results = pvd.nvs_single_view_1drc_iterative()

    elif opts.mode == 'single_view_nbv':
        all_results = pvd.nvs_single_view_nbv()