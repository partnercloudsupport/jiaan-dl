# # -*- coding: utf-8 -*-
"""
Lambda client
"""
import boto3, os, logging, json
from botocore.client import ClientError

class LambdaClient(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def audit_threat(self, msg):
        """Invoke Transaction Auditing service to track threats for analytics

        Args:
            msg (dict): payload

        Returns:
            dict
        """
        client = boto3.client('lambda')

        try:
            resp = client.invoke(
                FunctionName = os.getenv('EVENT_ROUTER', None),
                InvocationType = 'Event',
                Payload = bytes(json.dumps(msg))
            )
        except ClientError as ce:
            self.logger.error(ce)
            resp = None

        return resp