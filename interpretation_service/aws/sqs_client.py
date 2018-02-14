# # -*- coding: utf-8 -*-
"""
SNS Client
"""
import boto3, logging, os, json
from botocore.client import ClientError

class SqsClient(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARN)

    def notify_threat(self, msg):
        """Deliver default message to SNS Topic

        Args:
            msg (dict): payload

        Returns:
            dict
        """
        client = boto3.client('sns')

        try:
            resp = client.publish(
                TargetArn = os.getenv('SNS_TOPIC', None),
                Message = json.dumps({'default':json.dumps(msg)}),
                MessageStructure = 'json'
            )
        except ClientError as ce:
            self.logger.error(ce, exc_info=True)
            resp = None

        return resp