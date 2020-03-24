from run_squad import process_inputs, process_result, process_output
import grpc
import tensorflow as tf  
import json
import os
import requests
import tokenization
import grpc 
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

examples, features = process_inputs()

hostport = "192.168.0.27:8500"
headers = { "Content-type": "application/json" }

record_iterator = tf.python_io.tf_record_iterator(path='./eval.tf_record')

channel = grpc.insecure_channel(hostport)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

model_request = predict_pb2.PredictRequest()
model_request.model_spec.name = 'bert-qa'

all_results = []

for string_record in record_iterator:
    
    model_request.inputs['examples'].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                string_record,
                dtype=tf.string,
                shape=[8]
            )
    )
    
    result_future = stub.Predict.future(model_request, 30.0)  
    raw_result = result_future.result().outputs
    all_results.append(process_result(raw_result))

with open('test-file.json') as json_file:
    data = json.load(json_file)

result = process_output(all_results, examples, features, data['data'])
print(json.dumps(result))