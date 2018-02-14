# # -*- coding: utf-8 -*-
"""
Author: Matthew Pettit
Project: DeepLens Challenge - Jiaan
"""
import logging, json

from aws import LambdaClient, SqsClient
from deserializer import InferenceEvent

threat_description = {
    1 : 'Suspicious Behavior',
    2 : 'Threat Awareness',
    3 : 'Threat Detected'
}
"""obj:dict Describes threat level
"""

def calculate_threat(vectors):
    """Calculates the level of threat based on a targeted set of trained objects

    Args:
        vectors (list of dicts): Vectors and their respective label

    Returns:
        float
    """
    threat = 0.00

    for vector in vectors:
        label = vector['label'].replace('\n','')
        prob = vector['probability']

        if label in ['n09773778']:
            threat += prob
        elif label in ['n03219004', 'n05802552']:
            threat += prob / 2
        elif label in ['n01151256', 'n08531701', 'n09509557']:
            threat += prob / 4

    nom = int((threat / len(vectors)) * 10) if len(vectors) > 0 else 0

    if nom > 0:
        if nom < 3: return 1
        if nom < 6: return 2
        return 3

    return 0

def handler(event, context):
    """Event Handler for IoT
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(event)

    inference = InferenceEvent(event)

    threat_level = calculate_threat(inference.vectors)

    if threat_level > 0:
        try:
            audit_client = LambdaClient()
            notification_client = SqsClient()

            # Audit the inferred threat
            if os.getenv('EVENT_ROUTER', None) is not None:
                response = audit_client.audit_threat(
                    {
                        'source': 'localhost',
                        'level': threat_level,
                        'description': threat_description[threat_level]
                    })
    
                logger.info(response)
            else:
                logger.warn("Audit Service not implemented.")

            # Disseminate threat detection
            if os.getenv('SNS_TOPIC', None) is not None:
                response = notification_client.notify_threat(
                    {
                        'source': 'localhost',
                        'level': threat_level,
                        'description': threat_description[threat_level]
                    })
    
                logger.info(response)
            else:
                logger.warn("Notification Service not implemented.")
        except Exception as e:
            logger.error(e)