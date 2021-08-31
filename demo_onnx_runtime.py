import onnx
import onnxruntime
import numpy as np
import argparse
import os
import sys

def load_model(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    sess_options = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(model_path, sess_options)
    
    return sess, onnx_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', default='deployment/layoutlm_onnx/1/model.onnx')  
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        sys.exit()
    
    sess, onnx_model = load_model(args.model_dir)
    output_name = sess.get_outputs()[0].name
    
    # Get random inputs representing as such
    input_ids = np.random.randint(0, 1, (1, 512))
    bbox = np.random.randint(0, 1, (1, 512, 4))
    attention_mask = np.random.randint(0, 1, (1, 512))
    token_type_ids = np.random.randint(0, 1, (1, 512))
    
    preds = sess.run([output_name], {
        'input_ids':input_ids,
        'bbox': bbox,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    })[0]
    
    print(preds)
    print(preds.shape)