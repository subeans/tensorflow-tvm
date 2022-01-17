from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime

import tvm.testing 
import time
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--imgsize',default=224 , type=int)
parser.add_argument('--arch',default='arm' , type=str)
parser.add_argument('--export',default=False , type=bool)

args = parser.parse_args()

model_name = args.model
batch_size = args.batchsize
size = args.imgsize
arch_type = args.arch
export = args.export

def make_dataset(batch_size,size):
    image_shape = (size, size,3)
    # image_shape = (3,size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

if arch_type == "intel":
    target = "llvm"
else:
    target = tvm.target.arm_cpu()

ctx = tvm.cpu()
data,image_shape = make_dataset(batch_size,size)
shape_dict = {"input_1": data.shape}

load_model = time.time()
graph_fn = f"./{model_name}_{batch_size}_{arch_type}.json"
lib_fn = f"./{model_name}_{batch_size}_{arch_type}.tar"
params_fn = f"./{model_name}_{batch_size}_{arch_type}.params"

loaded_graph = open(graph_fn).read()
loaded_mod = tvm.runtime.load_module(lib_fn)
loaded_params = open(params_fn, "rb").read()

loaded_rt = tvm.contrib.graph_runtime.create(loaded_graph, loaded_mod, ctx)
loaded_rt.load_params(loaded_params)
print('load_model time', (((time.time() - load_model) ) * 1000),"ms")

#onnx_model = onnx.load(f'./{model_name}.onnx')
#module = graph_executor.GraphModule(loaded_lib["default"](ctx))
# module.set_input("input_1", data)
# module.run()
# out = module.get_output(0).asnumpy()

measurements = 5
iter_times = []
print("-"*10,"time.time Module","-"*10)
for i in range(measurements):
    start_time = time.time()
    loaded_rt.run(data = data)
    print(f"TVM {model_name}-{batch_size} inference_time : ",(time.time()-start_time)*1000,"ms")
    iter_times.append(time.time() - start_time)

print(f"TVM model {model_name}-{batch_size} runtime time elapsed",np.mean(iter_times) * 1000 ,"ms")
print("\n")



data_tvm = tvm.nd.array(data.astype('float32'))

e = loaded_rt.module.time_evaluator("run", ctx, number=5, repeat=1)
t = e(data_tvm).results
t = np.array(t) * 1000
   
print("="*10,"TVM time evaluator module","="*10)
print('{} (batch={}): {} ms'.format(model_name, batch_size, t.mean()))
