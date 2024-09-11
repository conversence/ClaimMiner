"""
CLI to apply tasks in bulk
"""

# Copyright Society Library and Conversence 2022-2024
import argparse
import asyncio
from logging import getLogger

from .. import dispatcher, set_log_config
from ..task_registry import get_task_map, TaskRegistry
from ..utils import get_sentry_sdk
from ..pyd_models import process_status

set_log_config("logging_cli")

sentry_sdk = get_sentry_sdk("cli")
logger = getLogger(__name__)


async def prepare():
    _ = await TaskRegistry.get_full_registry()
    from ..models import finalize_db_models

    await finalize_db_models()


async def apply_task(
    task_name=None,
    target_ids=None,
    targets_in_status=None,
    collection_name=None,
    delete_results=False,
    **kwargs,
):
    try:
        from ..bulk_tasks import bulk_apply_tasks

        await dispatcher.start()
        if not task_name:
            print("You must specify the subtask")
            return
        await bulk_apply_tasks(
            task_name,
            process_status[targets_in_status],
            collection_name,
            target_ids=target_ids,
            delete=delete_results,
            **kwargs,
        )
    except Exception as e:
        if sentry_sdk:
            sentry_sdk.capture_exception(e)
        logger.exception("", exc_info=e)
        import pdb

        pdb.post_mortem()
    finally:
        await dispatcher.stop()


if __name__ == "__main__":
    with asyncio.Runner() as runner:
        runner.run(prepare())
        parser = argparse.ArgumentParser()
        parser.add_argument("--collection-name", "-c")
        parser.add_argument("--delete-results", "-d", action="store_true")
        parser.add_argument("--target-ids", "-t", type=int, action="append")
        parser.add_argument(
            "--targets-in-status",
            "-s",
            choices=[t.name for t in process_status],
            default=process_status.not_requested.name,
        )
        subparsers = parser.add_subparsers()
        for task_name, task_cls in get_task_map().items():
            subparser = subparsers.add_parser(task_name)
            subparser.set_defaults(task_name=task_name)
            task_cls.add_parse_params(subparser)
        args = parser.parse_args()
        runner.run(apply_task(**vars(args)))
