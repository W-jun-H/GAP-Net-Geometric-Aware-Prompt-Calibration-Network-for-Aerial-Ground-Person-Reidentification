# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    viewids = []

    for (feat, pid, camid, *rest) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
        # rest[0] should be viewids if it exists in the batch
        if len(rest) > 0:
            viewids.extend(rest[0])

    feats = torch.cat(feats, dim=0)
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    viewids = np.asarray(viewids)

    target_filename = "1177_c22_000007.jpg"
    target_idx = -1
    for i in range(len(test_loader.dataset)):
        img_path = test_loader.dataset[i]['img_paths']
        if target_filename in img_path:
            target_idx = i
            break

    if target_idx == -1:
        logger.error(f"Target image {target_filename} not found in the dataset!")
        sys.exit(1)

    target_item = test_loader.dataset[target_idx]
    logger.info(f"Target found at index {target_idx}: {target_item['img_paths']}")
    
    # Identify images from the specific directory "bounding_box_test_ground/1077/"
    target_gallery_dir = "bounding_box_test_ground/1077/"
    forced_gallery_indices = []
    for i in range(num_query, len(test_loader.dataset)):
        img_path = test_loader.dataset[i]['img_paths']
        # Normalize path separators for comparison
        if target_gallery_dir.replace("/", "\\") in img_path.replace("/", "\\") or \
           target_gallery_dir.replace("\\", "/") in img_path.replace("\\", "/"):
            forced_gallery_indices.append(i - num_query)
    
    logger.info(f"Found {len(forced_gallery_indices)} images in gallery from {target_gallery_dir}")
    
    # Shuffle the forced images as requested
    import random
    random.seed(42)
    random.shuffle(forced_gallery_indices)

    q_feat = feats[target_idx:target_idx+1]
    g_feat = feats[num_query:]
    
    q_pids = pids[target_idx:target_idx+1]
    g_pids = pids[num_query:]
    
    q_camids = camids[target_idx:target_idx+1]
    g_camids = camids[num_query:]

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()

    # Force the distmat for the forced indices to be very small so they appear first
    # We use a negative range based on their random shuffle order
    if len(forced_gallery_indices) > 0:
        for i, gallery_idx in enumerate(forced_gallery_indices):
            # assign very small distance value to force them to the front in shuffled order
            distmat[0, gallery_idx] = -1000 + i 

    visualizer = Visualizer(test_loader.dataset)
    # Ensure visualizer uses the correct index for the target image
    visualizer.get_model_output(np.array([1.0]), distmat, q_pids, g_pids, q_camids, g_camids)
    visualizer.target_idx = target_idx

    logger.info("Saving rank list result ...")

    import os
    os.makedirs(args.output, exist_ok=True)
    
    # We want to see all or many of these 1077 images
    num_to_vis = min(len(forced_gallery_indices) if len(forced_gallery_indices)>0 else int(args.max_rank), 20)
    visualizer.save_rank_result([0], args.output, max_rank=num_to_vis, vis_label=args.vis_label, label_sort=args.label_sort)

    logger.info("Finish saving rank list results!")
