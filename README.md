# LayoutLM - Deployment
Repo to test conversion of SOTA layoutlm model conversion and deployment

# How it works
- To convert for trition usage, the model file must be either one of 2 formats in this case the base model is in pytorch, to use it in deployment, it must first be in this format:
    - Libtorch
    - ONNX

This tutorial focuses on converting it to onnx, and it will use on the SEQ Labelling Task, the other will be on hold for now

NOTE: The tokenizer that comes with it cannot be convert to onnx, therefore it has to be seperated into 2 components

# Model Download
TBA

# How to run
## Run using triton inference server
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

## Run using onnxruntime
The other option is to test directly with a python script

Note: This option uses random to randomly generate the numbers therefore it does not work very well
1) First download or convert the model to onnx

2) Run the following python script
```bash
python demo_onnx_runtime.py
```

## TODO
- [x] Data parser
- [x] Create inference script for demo for script
- [x] Convert model from torch to onnx
- [x] Load model to trition inference server
- [x] Write triton inference script for deployment
- [x] Write onnxruntime infer script

## TODO Clean up
- [x] Write py file for conversion
- [x] Write py file for deployment

## Future Improvement
- [ ] Wrap up in a fastapi/flask for more complete repo
- [ ] Classification Model to ONNX
- [ ] Set to deploy in trition

# Reference
- You can check the original REPO code here
    - https://github.com/microsoft/unilm/tree/master/layoutlm/deprecated