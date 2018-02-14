# Jiaan - Intelligent Security Camera

Jiaan is an intelligent security camera built for the DeepLens challenge. The services target DeepLens- and IoT-specific transactions that process, interpret, and act on classified objects detected by the SSD r-CNN framework. The model has been trained from the VGG16 Reduced model using a custom data set that is not included in this repository as they are too large (~2.9GB).

## Installation

```python
pip install boto3 cv2 mxnet
```

** DeepLens hardware is required | awscam and greengrass packages are required

## History

v0.1.0 - DeepLens challenge submission

## License

MIT