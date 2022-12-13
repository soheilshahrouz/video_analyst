# -*- coding: utf-8 -*-

import argparse
import os.path as osp

from loguru import logger

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder

from videoanalyst.model.task_model.taskmodel_impl.siamese_track import SiamFCppTemplateMaker, SiamFCppForward

def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    return parser


def build_siamfcpp_tester(task_cfg):
    # build model
    model = model_builder.build("track", task_cfg.model)

    temp_model = SiamFCppTemplateMaker(model).cuda().eval()

    dummy_input = torch.randn(1, 127, 127, 3, device="cuda")
    input_names  = [ "template_maker_input" ]
    output_names = [ "template_maker_kernel_output", "template_maker_reg_output", "template_maker_cls_output" ]

    torch.onnx.export(temp_model, dummy_input, "SiamFCpp_Template_Maker.onnx", export_params=True,
        verbose=True, input_names=input_names, output_names=output_names)


    

    dummy_input = torch.randn(1, 303, 303, 3, device="cuda")
    dummy_reg   = torch.randn(1, 256,   4, 4, device="cuda")
    dummy_cls   = torch.randn(1, 256,   4, 4, device="cuda")
    simp_forw_model = SiamFCppForward(model).cuda()

    input_names  = [ "forward_input, forward_reg, forward_cls" ]
    output_names = [ "forward_delta0_output", "forward_delta1_output", "forward_delta2_output", "forward_delta3_output", "forward_cls_output" ]

    torch.onnx.export(simp_forw_model, (dummy_input, dummy_reg, dummy_cls), "SiamFCpp_Forward.onnx", export_params=True,
        verbose=True, input_names=input_names, output_names=output_names)


    exit()
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers


def build_sat_tester(task_cfg):
    # build model
    tracker_model = model_builder.build("track", task_cfg.tracker_model)
    tracker = pipeline_builder.build("track",
                                     task_cfg.tracker_pipeline,
                                     model=tracker_model)
    segmenter = model_builder.build('vos', task_cfg.segmenter)
    # build pipeline
    pipeline = pipeline_builder.build('vos',
                                      task_cfg.pipeline,
                                      segmenter=segmenter,
                                      tracker=tracker)
    # build tester
    testers = tester_builder('vos', task_cfg.tester, "tester", pipeline)
    return testers


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    if task == 'track':
        testers = build_siamfcpp_tester(task_cfg)
    elif task == 'vos':
        testers = build_sat_tester(task_cfg)
    for tester in testers:
        tester.test()
