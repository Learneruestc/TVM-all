# -*- coding: utf-8 -*-

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-relay-quick-start:

Quick Start Tutorial for Compiling Deep Learning Models
=======================================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Truman Tian <https://github.com/SiNZeRo>`_

This example shows how to build a neural network with Relay python frontend and
generates a runtime library for Nvidia GPU with TVM.
Notice that you need to build TVM with cuda and llvm enabled.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

######################################################################
# Overview for Supported Hardware Backend of TVM
# ----------------------------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tvm_support_list.png
#      :align: center
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import Relay and TVM.

import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
# from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
# import numpy as np
# import tvm.relay as relay

######################################################################
# Define Neural Network in Relay
# ------------------------------
# First, let's define a neural network with relay python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in Relay.
# Parameters are initialized with Xavier initializer.
# Relay also supports other model formats such as MXNet, CoreML, ONNX and
# Tensorflow.
#
# In this tutorial, we assume we will do inference on our device and
# the batch size is set to be 1. Input images are RGB color images of
# size 224 * 224. We can call the
# :py:meth:`tvm.relay.expr.TupleWrapper.astext()` to show the network
# structure.

# batch_size = 1
# num_class = 1000
# image_shape = (3, 224, 224)
# data_shape = (batch_size,) + image_shape
# out_shape = (batch_size, num_class)
# mod, params = relay.testing.resnet.get_workload(
#     num_layers=18, batch_size=batch_size, image_shape=image_shape
# )

#加载onnx预训练模型
model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# Seed numpy's RNG to get consistent results
np.random.seed(0)

#加载一张测试图片 - 经典的猫图片
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

#Import the graph to Relay
# 将onnx 图转换为Relay图，Input name是生成的.onnx模型的输入代号
# The input name may vary across model types. You can use a tool
# like Netron to check input names
target = "llvm"
input_name = "data"
shape_dict = {input_name: img_data.shape}


mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)


# set show_meta_data=True if you want to show meta data
# print(mod.astext(show_meta_data=False))

######################################################################
# Compilation
# -----------
# Next step is to compile the model using the Relay/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 3. The optimization passes include
# operator fusion, pre-computation, layout transformation and so on.
#
# :py:func:`relay.build` returns three components: the execution graph in
# json format, the TVM module library of compiled functions specifically
# for this graph on the target hardware, and the parameter blobs of
# the model. During the compilation, Relay does the graph-level
# optimization while TVM does the tensor-level optimization, resulting
# in an optimized runtime module for model serving.
#
# We'll first compile for Nvidia GPU. Behind the scene, :py:func:`relay.build`
# first does a number of graph-level optimizations, e.g. pruning, fusing, etc.,
# then registers the operators (i.e. the nodes of the optimized graphs) to
# TVM implementations to generate a `tvm.module`.
# To generate the module library, TVM will first transfer the high level IR
# into the lower intrinsic IR of the specified target backend, which is CUDA
# in this example. Then the machine code will be generated as the module library.

opt_level = 3
# target = tvm.target.cuda()
# target = tvm.target.create('llvm')
target = "llvm"
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

#####################################################################
# Run the generate library
# ------------------------
# Now we can create graph executor and run the module on Nvidia GPU.

# create random input
# dev = tvm.cuda()
dev = tvm.cpu()
# data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
module.set_input("data", img_data)
# run
module.run()
# get output
# out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# # Print first 10 elements of output
# print(out.flatten()[0:10])

output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()


from scipy.special import softmax

# Download a list of labels
labels_path = "../TVM1/synset.txt"

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1] #逆序排列，输出索引是从大到小的
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files and load them
# back in deploy environment.

####################################################

# save the graph, lib and params into separate files
from tvm.contrib import utils

temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")

lib.export_library(path_lib)
print(temp.listdir())

####################################################

# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(img_data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.set_input("data", img_data)
# run
module.run()
output_shape = (1, 1000)
tvm_output1 = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

scores = softmax(tvm_output1)
scores1 = np.squeeze(scores)
ranks = np.argsort(scores1)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores1[rank]))

# check whether the output from deployed module is consistent with original one
# tvm.testing.assert_allclose(out_deploy, out, atol=1e-5)

