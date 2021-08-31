import numpy as np
import tritonclient.http as httpclient

from transformers import AutoTokenizer
from layoutlm.data.funsd import read_examples_from_file, convert_examples_to_features
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import utils
import argparse

def get_triton_input_output_shape():
    inputs= [
        httpclient.InferInput(
            "input_ids",
            [1, 512],
            "INT64"
        ),
        httpclient.InferInput(
            "attention_mask",
            [1, 512],
            "INT64"
        ),
        httpclient.InferInput(
            "token_type_ids",
            [1, 512],
            "INT64"
        ),
        httpclient.InferInput(
            "bbox",
            [1, 512, 4],
            "INT64"
        ),
    ]

    outputs = [
        httpclient.InferRequestedOutput('output', binary_data=True)
    ]
    
    return inputs, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', default='model/layoutlm-base-uncased/')
    parser.add_argument('-l', '--label_dir', default='data/infer_data/labels.txt')
    parser.add_argument('-d', '--data_dir', default='data/infer_data/')
    
    args = parser.parse_args()
    
    client = httpclient.InferenceServerClient('localhost:8000')
    
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
    
    # Set values to send to triton server
    inputs, outputs = get_triton_input_output_shape()
    inputs[0].set_data_from_numpy(all_input_ids[0].unsqueeze(0).numpy())
    inputs[1].set_data_from_numpy(all_input_mask[0].unsqueeze(0).numpy())
    inputs[2].set_data_from_numpy(all_segment_ids[0].unsqueeze(0).numpy())
    inputs[3].set_data_from_numpy(all_bboxes[0].unsqueeze(0).numpy())
    
    # Run inference to triton server
    results = client.infer(
        'layoutlm_onnx',inputs, outputs=outputs,    
    )
    
    # Get result - must specify output name, check pbtxt for more info
    preds = results.as_numpy('output')[0]
    preds = np.argmax(preds, axis=1)
    
    print(preds)