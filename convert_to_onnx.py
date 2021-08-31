from layoutlm import FunsdDataset, LayoutlmConfig, LayoutlmForTokenClassification
from transformers import BertTokenizer, AutoTokenizer
from layoutlm.data.funsd import read_examples_from_file, convert_examples_to_features

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import utils
import argparse
import os

# Create LayoutLM with a new forward pass where i return a single tensor, instead of a tuple
class LayoutLM(nn.Module):
    def __init__(self, model):
        super(LayoutLM, self).__init__()
        self.model = model
        
    def forward(self, input_ids, bbox, attention_mask, token_type_ids):
        return self.model(
            input_ids, bbox, attention_mask, token_type_ids
        )[0]
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_dir', default='data/infer_data/labels.txt')
    parser.add_argument('-d', '--data_dir', default='data/infer_data/')
    parser.add_argument('-m', '--model_dir', default='model/layoutlm-base-uncased/')
    parser.add_argument('-o', '--output_dir', default='deployment/layoutlm_onnx/1/model.onnx')
    
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    labels = utils.get_labels(args.label_dir)
    num_labels = len(labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    
    examples = read_examples_from_file(args.data_dir, 'test')
    features = convert_examples_to_features(
        examples, labels, 512,
        tokenizer, cls_token_at_end=False, # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token, cls_token_segment_id=0,
        sep_token=tokenizer.sep_token, sep_token_extra=False,
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id,
    )
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_ids for f in features], dtype=torch.long
    )
    all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)
    
    model = LayoutlmForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    
    ## Prepare necessary inputs here
    layout_model = LayoutLM(model)
    layout_model.to(device)
    layout_model.eval()
    
    inputs = {
        'input_ids': all_input_ids[0].unsqueeze(0).to(device),
        'attention_mask': all_input_mask[0].unsqueeze(0).to(device),
        'token_type_ids': all_segment_ids[0].unsqueeze(0).to(device),
        'bbox': all_bboxes[0].unsqueeze(0).to(device)
    }
    
    sample_inputs = (
        inputs['input_ids'], inputs['bbox'], 
        inputs['attention_mask'], inputs['token_type_ids'], 
    )
    torch.onnx.export(
        layout_model, sample_inputs, args.output_dir, export_params=True,
        opset_version=11, do_constant_folding=True,
        input_names=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'], output_names=['output'],
        dynamic_axes={
            'input_ids' : {0 : 'batch_size'},    # variable lenght axes
            'bbox' : {0 : 'batch_size'},
            'attention_mask' : {0 : 'batch_size'},
            'token_type_ids' : {0 : 'batch_size'},
            'output' : {0 : 'batch_size'}
        }
    )
    
    if os.path.exists(args.output_dir):
        print("Model Onnx Created")
    else:
        print('Model Onnx is not created')