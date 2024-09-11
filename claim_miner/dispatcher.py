from abc import ABC, abstractmethod


class Dispatcher(ABC):
    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def flush(self):
        pass

    @abstractmethod
    async def trigger_task(self, task_name: str, **kwargs):
        pass
