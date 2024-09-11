"""
Utility functions to define the set of kafka topics and obtaining consumer and producers for those topics.
"""

# Copyright Society Library and Conversence 2022-2024
from logging import getLogger
import aiokafka

import orjson as json

from . import config
from .dispatcher import Dispatcher
from .task_registry import get_task_map

logger = getLogger(__name__)
CONSUMER = None
PRODUCER = None
TOPICS = None


def get_topics():
    global TOPICS
    if TOPICS is None:
        TOPICS = list(get_task_map().keys())
    return TOPICS


def serializer(j):
    return json.dumps(j) if j else None


def deserializer(s):
    return json.loads(s) if s else None


async def get_consumer():
    global CONSUMER
    if CONSUMER is None:
        CONSUMER = aiokafka.AIOKafkaConsumer(
            *get_topics(),
            bootstrap_servers=f"{config.get('kafka', 'host', fallback='localhost')}:{config.get('kafka', 'port', fallback=9092)}",
            value_deserializer=deserializer,
            key_deserializer=deserializer,
            group_id="ClaimMiner",
        )
        await CONSUMER.start()
        logger.info("Consumer ready")
    return CONSUMER


async def get_producer():
    global PRODUCER
    if PRODUCER is None:
        PRODUCER = aiokafka.AIOKafkaProducer(
            bootstrap_servers=f"{config.get('kafka', 'host', fallback='localhost')}:{config.get('kafka', 'port', fallback=9092)}",
            value_serializer=serializer,
            key_serializer=serializer,
        )
        await PRODUCER.start()
        logger.info("Producer ready")
    return PRODUCER


async def stop_producer():
    global PRODUCER
    if PRODUCER is not None:
        await PRODUCER.stop()
        logger.info("Producer stopped")
    PRODUCER = None


async def stop_consumer():
    global CONSUMER
    if CONSUMER is not None:
        await CONSUMER.stop()
        logger.info("Consumer stopped")
    CONSUMER = None


class TopicWrapper:
    def __init__(self, topic):
        self.topic = topic

    async def send_and_wait(self, value, key=None):
        producer = await get_producer()
        await producer.send_and_wait(self.topic, value, key)

    async def send_soon(self, value, key=None):
        producer = await get_producer()
        await producer.send(self.topic, value, key)


wrappers = {}


def get_channel(topic):
    global wrappers
    if topic not in wrappers:
        wrappers[topic] = TopicWrapper(topic)
    return wrappers[topic]


class KafkaDispatcher(Dispatcher):
    async def start(self):
        await get_producer()

    async def stop(self):
        await stop_producer()

    async def flush(self):
        global PRODUCER
        if PRODUCER:
            await PRODUCER.flush()

    async def trigger_task(self, task_name, **kwargs):
        task_cls = get_task_map().get(task_name)
        assert task_cls
        # Task creation may fail early
        if task_cls.materialize_analysis and (analysis_id := kwargs.get("analysis_id")):
            val = dict(analysis_id=analysis_id)
            key = str(analysis_id)
        else:
            task = task_cls(**kwargs)
            val = task.as_dict()
            key = task.get_channel_key()
        await get_channel(task_cls.name).send_soon(val, key=key)
