import pandas as pd
from transformers import (BertTokenizer, 
                          BertForSequenceClassification,)
import os
import sys
import logging
import json

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def model_fn(model_dir):
    pytorch_model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return (pytorch_model, tokenizer)

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/json':
        payload = json.loads(request_body)
        data = list(str(elem) for elem in payload)
        print(f"data: {data}")
        return data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        print("BAD NEWS: Unsupported input type")
        pass

def predict_fn(input_data, model_token_tuple):
    print(input_data)
    model, tokenizer = model_token_tuple
    
    predictions = []
    for item in input_data:
        tokenized_item = tokenizer.encode_plus(item, add_special_tokens=True, return_tensors='pt')
        pred = model(tokenized_item['input_ids'], 
                     token_type_ids=tokenized_item['token_type_ids'])[0].argmax().item()
        predictions.append(pred)

    print(f"prediction: {predictions}")
    return {'predictions': predictions}

    
    
def output_fn(prediction_output, accept='application/json'):
    classes = {0: 'Lame', 1: 'Correct'}
    
    print(prediction_output)
    
    result = [classes[pred] for pred in prediction_output['predictions']]
    print(result)
    
    if accept == 'application/json':
        return json.dumps(result), accept
    
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')
    