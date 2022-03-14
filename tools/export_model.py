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
import sys
import random
import paddle
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from config import get_arguments
from model.training import *
from model.imresize import imresize
import model.functions as functions
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=1):
    """
    利用已训练的模型, 生成新图像
    """
    if in_s is None:
        in_s = paddle.full(reals[0].shape, 0)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m_zeropad = ZeroPad2d(padding=[int(pad1),int(pad1),int(pad1),int(pad1)])
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy])
                z_curr = paddle.expand(z_curr, shape = [1,3,z_curr.shape[2],z_curr.shape[3]])
                #z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m_zeropad(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy])
                z_curr = m_zeropad(z_curr)

            if images_prev == []:
                I_prev = m_zeropad(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m_zeropad(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m_zeropad(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt
            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)
            return G

    
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument(
        '--save_inference_dir', default='./infer/', help='path where to save')
    parser.add_argument('--input_dir', help='input image dir', default='../Input/Images')
    parser.add_argument('--input_name', help='input image name',default='colusseum.png')
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='random_samples_arbitrary_sizes') #, required=True
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=2)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        if opt.mode == 'random_samples':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = load_trained_pyramid_New(opt)
            in_s = functions.generate_in2coarsest(reals,1,1,opt)
            test_model = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)

        elif opt.mode == 'random_samples_arbitrary_sizes':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = load_trained_pyramid_New(opt)
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
            test_model = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h)
    
    test_model.eval()

    # decorate model with jit.save
    model = paddle.jit.to_static(
        test_model,
        input_spec=[
            InputSpec(shape=[-1, 3, -1, -1], dtype='float32', name='x'),
            InputSpec(shape=[-1, 3, -1, -1], dtype='float32', name='y')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(opt.save_inference_dir, "inference"))
    print(f"inference model has been saved into {opt.save_inference_dir}")



