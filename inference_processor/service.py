# # -*- coding: utf-8 -*-
"""
Author: Matthew Pettit
Project: DeepLens Project - Jiaan
"""
import os, logging
from threading import Timer, Thread

import greengrasssdk, awscam, cv2

from model import InferEvent, Vector

client = greengrasssdk.client('iot-data')

iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame)

class FIFO_Thread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"

        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)

        f = open(fifo_path,'w')

        while True:
            try:
                f.write(jpeg.tobytes())
            except IOError:
                continue

def greengrass_infinite_infer_run():
    try:
        # TODO: Parameterize model name
        model_name = 'deploy_ssd_resnet50_300'
        model_path = "/opt/awscam/artifacts/mxnet_{}_FP16_FUSED.xml".format(model_name)
        model_type = "ssd"

        # Trained image shape
        input_width = 300
        input_height = 300

        with open('docs/synset.txt', 'r') as _f:
            labels = dict(enumerate(_f, 1))

        client.publish(topic=iotTopic, payload="Labels loaded")

        # Start thread to write frames to disk
        results_thread = FIFO_Thread()
        results_thread.start()

        # MxNet Model
        model = awscam.Model(model_path, {"GPU": 1})

        client.publish(topic=iotTopic, payload="Model loaded")

        while True:
            ret, frame = awscam.getLastFrame()
            if ret == False:
                raise Exception("Failed to get frame from the stream")

            inferred_obj = InferEvent(frame, model_name, model_type)

            frameResize = cv2.resize(frame, (input_width, input_height))

            # Run inference
            output = model.doInference(frameResize)
            parsed_results = model.parseResult(model_type, output)['ssd']

            # Parse and document vectors
            for obj in parsed_results:
                if obj['prob'] > 0.5:
                    vector = Vector((obj['xmin'], obj['xmax'], obj['ymin'], obj['ymax'],), inferred_obj.xscale, inferred_obj.yscale)
                    inferred_obj.add_vector(vector, (labels[obj['label']], obj['prob'],))

                    cv2.rectangle(frame, (vector.xmin, vector.ymin), (vector.xmax, vector.ymax), (255, 165, 20), 4)
                    cv2.putText(frame, '{}: {:.2f}'.format(labels[obj['label']], obj['prob']), (vector.xmin, vector.ymin-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 20), 4)

            # Trigger event pipeline
            client.publish(topic=iotTopic, payload = str(inferred_obj))

            # Update output
            global jpeg
            ret,jpeg = cv2.imencode('.jpg', frame)
    except Exception as e:
        client.publish(topic=iotTopic, payload=str(e))

# Invoke inference processing
greengrass_infinite_infer_run()

def handler(event, context):
    return