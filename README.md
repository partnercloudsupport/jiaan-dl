# Jiaan - Intelligent Security Camera

Jiaan is an intelligent security camera built for the DeepLens challenge. The services target DeepLens- and IoT-specific transactions that process, interpret, and act on classified objects detected by the [SSD](https://arxiv.org/abs/1512.02325) r-CNN framework. The model has been trained from the VGG16 Reduced model using a custom data set that is not included in this repository as they are too large (~2.9GB).

## Model
- Trained and optimized model is located under models/
- Original VGG16 Reduced model Jiaan was fine tuned on is located in /jiaan-train/vgg16_reduced

## Installation

### Inference Processing and Interpretation Services
```python
pip install boto3 cv2 mxnet
```

+ Deploy /inference_processor to Lambda
    - Updated [model_name] with any SSD r-CNN trained model deployed to DeepLens
    - Update the docs/synset.txt with all labels available for inference
+ Deploy /interpretation_service to Lambda
    - Create SNS Topic for test notifications and subscribe
    - Deploy your own auditing service (preferrably using ElasticSearch)
    - Add SNS and Lambda ARNs for notification and auditing services
+ Create new model for DeepLens using externally trained model
+ Create new project with model and add the inference_processor Lambda function
+ Deploy project to DeepLens

### Model Training

+ Import Jupyter Notebook project into Sagemaker
+ Update account-specific paths with your own S3 buckets

## Testing

```python
cd inference_processor
python -m unittest tests/
```

** DeepLens hardware is required | awscam and greengrass packages are required

## External Dependencies

- Greengrass SDK
- awscam

## History

v0.1.0 - DeepLens challenge submission

## License

MIT