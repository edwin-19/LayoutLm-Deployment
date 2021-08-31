# LayoutLM - Deployment
Repo to test conversion of SOTA layoutlm model conversion and deployment

# Installation
1) Clone the repo
```bash
git clone https://github.com/edwin-19/LayoutLm-Deployment.git
```

2) Download and preprocess the data
```bash
./preprocess.sh
```

3) Convert available model to onnx
```bash
python convert_to_onnx.py
```
Note here you can move over to the triton client by specifying the specific path

4) Run Trition Server
- Remember to change the -v to point to the path of your triton server
- When using volume, you have to specify absolute path
```bash
# Run on CPU
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/mnt/882eb9b0-1111-4b8f-bfc3-bb89bc24c050/pytorch/layoutlm/deployment:/model nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --model-repository=/model 

# Or with GPU
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/mnt/882eb9b0-1111-4b8f-bfc3-bb89bc24c050/pytorch/layoutlm/deployment:/model nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --model-repository=/model 
```

5) Run test script
- Runs and specify localhost:8000
```bash
python test_trion_client.py
```


## TODO
- [x] Data parser
- [x] Create inference script for demo for script
- [x] Convert model from torch to onnx
- [x] Load model to trition inference server
- [x] Write triton inference script for deployment
- [ ] Write onnxruntime infer script

## TODO Clean up
- [x] Write py file for conversion
- [x] Write py file for deployment