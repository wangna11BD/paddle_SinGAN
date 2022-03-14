# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

import sys
import random
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor_x, self.input_tensor_y, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor_x = predictor.get_input_handle(input_names[0])
        input_tensor_y = predictor.get_input_handle(input_names[1])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor_x, input_tensor_y, output_tensor

    def run(self, x, y):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensor_x.copy_from_cpu(x)
        self.input_tensor_y.copy_from_cpu(y)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle sinGAN infer", add_help=add_help)

    parser.add_argument(
        "--model-dir", default='./infer/', help="inference model dir")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--use_gpu", default=True, type=bool, help="use gpu")

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    """
    inference_engine = InferenceEngine(args)

    # preprocess

    x = np.random.randn(args.batch_size, 3, 36, 78).astype("float32")
    y = np.random.randn(args.batch_size, 3, 36, 78).astype("float32")
    output = inference_engine.run(x, y)

    return output


if __name__ == "__main__":
    args = get_args()
    output = infer_main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("out", np.array([output]))
    reprod_logger.save("output_inference_engine.npy")