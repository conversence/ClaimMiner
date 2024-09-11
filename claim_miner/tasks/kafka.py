"""
The main worker loop for asynchronous events. Dispatches kafka messages to various tasks.
"""

# Copyright Society Library and Conversence 2022-2024
import asyncio
import atexit
import traceback
import argparse
import os
from logging import getLogger

from anyio import create_task_group

from .. import dispatcher, set_log_config
from ..kafka import get_consumer, get_producer, stop_consumer, stop_producer
from ..task_registry import get_task_map, TaskRegistry
from ..utils import get_sentry_sdk

set_log_config("event_logging")

RUNNING = True
DEBUGGING = False

logger = getLogger(__name__)
sentry_sdk = get_sentry_sdk("tasks")


async def kafka_listener(queue):
    global RUNNING
    consumer = await get_consumer()
    producer = await get_producer()
    task_map = get_task_map()
    logger.info("Consumer ready")
    async for msg in consumer:
        if not RUNNING:
            break
        logger.info("received %s %s", msg.topic, msg.value)
        try:
            assert msg.topic in task_map
            if analysis_id := msg.value.get("analysis_id"):
                task = await TaskRegistry.get_registry().task_from_analysis_id(
                    int(analysis_id)
                )
            else:
                task = task_map[msg.topic](**msg.value)
            await queue.put(task)
        except Exception as e:
            if sentry_sdk:
                sentry_sdk.capture_exception(e)
            traceback.print_exception(e)


def exit_handler():
    global RUNNING
    RUNNING = False


atexit.register(exit_handler)


async def finish():
    logger.info("Closing the loop")
    await stop_consumer()
    await stop_producer()


async def wrapper(task, *, task_status=None):
    logger.info(f"starting {task}")
    registry = TaskRegistry.get_registry()
    try:
        if task.analysis.id:
            await task.before_run()
        await task.run()
        logger.info(f"ended {task}")
        await registry.trigger_task_on_task_end(task)
        await registry.handle_created_objects()
        await dispatcher.flush()
    except Exception as e:
        if DEBUGGING:
            logger.exception("", exc_info=e)
            import pdb

            pdb.post_mortem(e.__traceback__)
        if sentry_sdk:
            sentry_sdk.capture_exception(e)
        logger.exception(f"Task {task} failed", exc_info=e)
        await registry.trigger_task_on_task_error(task)
    finally:
        task_status.started()


async def worker(queue, tg):
    while True:
        task = await queue.get()
        await tg.start(wrapper, task)


async def main(pool_size, restart_ongoing=True, debug=False):
    global DEBUGGING
    DEBUGGING = debug
    try:
        await TaskRegistry.get_full_registry()
        from ..models import finalize_db_models

        await finalize_db_models()
        queue = asyncio.Queue(pool_size)
        async with create_task_group() as tg:
            tg.start_soon(kafka_listener, queue)
            if restart_ongoing:
                from .. import Session, select
                from ..models import Analysis, process_status

                async with Session() as session:
                    ongoing_task_ids = list(
                        await session.scalars(
                            select(Analysis.id).filter_by(status=process_status.ongoing)
                        )
                    )
                for analysis_id in ongoing_task_ids:
                    task = await TaskRegistry.get_registry().task_from_analysis_id(
                        analysis_id
                    )
                    await task.schedule()
            for _ in range(pool_size):
                tg.start_soon(worker, queue, tg)
    finally:
        await finish()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--pool-size", type=int, default=4)
    arg_parser.add_argument("--debug", action="store_true")
    arg_parser.add_argument("-r", "--restart-ongoing", action="store_true")
    args = arg_parser.parse_args()
    # TODO: Find a way to re-load the session with a larger pool size? set SQLALCHEMY_POOL_SIZE for now.
    pool_size = args.pool_size
    if int(os.environ.get("SQLALCHEMY_POOL_SIZE", 5)) < pool_size:
        print(
            "Set the SQLALCHEMY_POOL_SIZE environment variable to at least the pool size"
        )
        exit(1)
    asyncio.run(main(pool_size, args.restart_ongoing, args.debug))
