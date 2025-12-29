<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->
# Remote Procedure Call
Remote procedure call (RPC) is a very important and useful feature of Apache TVM, it allows us to run compiled NN models on the real hardware without need to touch the remote device, the output result will be passed back automatically through network. By eliminating that manual work like, dumping input data to file, copying the exported NN model to remote device, setuping the device user environment, copying the output result to host development environment, RPC improve the development efficiency extremely.

RPC is very helpful in below 2 situations

- Hardware resources are limited

  RPC’s queue and resource management mechanism can make the hardware devices serve many developers and test jobs to run the compiled NN models correctly.

- Early-stage end to end evaluation

  Except the compiled NN model, all other parts are executed on the host development environment, so the complex preprocess or postprocess can be implemented easily.

## Suggested Architecture
Apache TVM RPC contains 3 tools, RPC tracker, RPC proxy, and PRC server. The RPC server is the necessary one, an RPC system can work correctly without RPC proxy and RPC tracker. RPC proxy is needed when you can’t access the RPC server directly. RPC tracker is strongly suggested to be added in your RPC system, because it provides many useful features, e.g., queue capability, multiple RPC servers management, manage RPC server through key instead of IP address.

![Suggested Architecture of RPC System](images/rpc_server_arch.svg)

As above figure shown, because there aren’t physical connection channels between machine A and machine C, D, so we set up a RPC proxy on machine B. The RPC tracker manage a request queue per RPC key, each user can request an RPC server from RPC tracker by a RPC key at anytime, if there is a idle RPC server with the same RPC key, then RPC tracker assign the RPC server to the user, if there isn’t a idle RPC server for the moment, the request will be put into the request queue of that RPC key, and check for it later.

## Setup

RPC tracker and RPC proxy don’t depend on any components or changes of the Zhouyi Compass Apache TVM, so the only work need to do for setting up them is executing below commands on the corresponding machine after installing Apache TVM according to the official document [**https://tvm.apache.org/docs/install/index.html**](https://tvm.apache.org/docs/install/index.html).

- RPC tracker
```shell
python3 -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190 --port-end 9191
```

- RPC proxy
```shell
python3 -m tvm.exec.rpc_proxy --host 0.0.0.0 --port 9090 --port-end 9091 --tracker RPC_TRACKER_IP:9190
```

RPC server depends on the Zhouyi NPU driver and the enhanced Apache TVM runtime, so the setting up work contain 2 parts, one is compiling the Zhouyi NPU driver correctly for your concrete device, the other one is constructing Apache TVM environment using the Zhouyi Compass Apache TVM package.

Please refer to the Zhouyi NPU driver relevant documents for the detailed compiling guide, and ensure the needed files `libaipudrv.so` and `aipu.ko` are generated successfully.

We have shipped prebuilt Apache TVM runtime of several supported platform in our WHL file, currently contains *`x86_64-linux-gnu`* and *`aarch64-linux-gnu`*. Apache TVM environment that RPC server can work is constructed by below commands, please modify *`aarch64-linux-gnu`* according to your concrete device.
```shell
unzip -d python dlhc-xxx-xxx.whl
rm python/tvm/*.so*
cp python/tvm/lib/aarch64-linux-gnu/libtvm_runtime.so python/tvm/
cp /xxx/libaipudrv.so python/tvm
cp /xxx/aipu.ko python/tvm
zip -r python.zip python
```
Then copy the compress package *`python.zip`* to your concrete device, and start the RPC server through below commands, please modify the *`RPC_PROXY_IP`*, *`RPC_PROXY_PORT`*, and *`RPC_KEY`* according to your concrete environment.
```shell
unzip python.zip
insmod python/tvm/aipu.ko
export PYTHONPATH=`pwd`/python:${PYTHONPATH}
python3 -m tvm.exec.rpc_server --host RPC_PROXY_IP --port RPC_PROXY_PORT --through-proxy --key RPC_KEY
```

## Working with RPC

Along with the cross compilation, user can execute the compiled NN model on remote device easily through RPC, because all other parts are run on host development environment, so any Python packages can be used to do the preprocess and postprocess works. The simple demo code of working with RPC is something like below.

```python
import numpy as np
from tvm import rpc
from tvm.compass.relax import Compass

# 1. Create Compass instance and set configurations.
compass = Compass("tf_mobilenet_v2.cfg")
# 2. Compile the nn model.
deployable = compass.compile(target="llvm -mtriple=aarch64-linux-gnu")

# 3. Run the nn model.
rpc_sess = rpc.connect_tracker(RPC_TRACKER_IP, RPC_TRACKER_PORT).request(RPC_KEY)
device_compiler = "…/bin/aarch64-linux-gnu-g++"
ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)
image = np.load("input.npy")
outputs = ee.run(image)
# 4. Check result.
predictions = outputs[0].numpy()[0]
top3_idxes = np.argsort(predictions)[-3:][::-1]
synset = get_imagenet_synset(1001)
print(f"Top3 labels: {[synset[idx] for idx in top3_idxes]}")
print(f"Top3 predictions: {[predictions[idx] for idx in top3_idxes]}")
```
The only difference between run on simulator and run on RPC contain 3 places, the first one is specifying device triple in target string, the others are specifying RPC session and cross compiler. For the complete and more examples, see the out-of-the-box use case in the delivered software package.
