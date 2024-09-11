from typing import Mapping, Dict, List, Self
from collections import defaultdict

from claim_miner.dispatcher import Dispatcher


class MockDispatcher(Dispatcher):
    def __init__(self: Self):
        self.channels: Mapping[str, List[Dict]] = defaultdict(list)

    async def start(self):
        pass

    async def stop(self):
        pass

    def clear(self):
        self.channels = defaultdict(list)

    async def flush(self):
        pass

    async def apply_tasks(self, until_empty=False):
        from claim_miner.task_registry import TaskRegistry

        registry = TaskRegistry.get_registry()
        first_loop = True
        while self.num_tasks() and (first_loop or until_empty):
            first_loop = False
            channels = self.channels
            self.clear()
            for channel, task_data in channels.items():
                for data in task_data:
                    if analysis_id := data.get("analysis_id"):
                        task = await registry.task_from_analysis_id(int(analysis_id))
                    else:
                        task = registry.task_by_name[channel](**data)
                    if analysis_id:
                        await task.before_run()
                    await task.run()
                    await registry.trigger_task_on_task_end(task)
                    await registry.handle_created_objects()

    async def trigger_task(self, task_name: str, **kwargs):
        self.channels[task_name].append(kwargs)

    def num_tasks(self):
        return sum(len(c) for c in self.channels.values())
