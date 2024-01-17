from abc import ABC, abstractmethod
from typing import Any

class Runnable(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.first: 'Runnable' = self
        self.next: 'Runnable' = None
        self.prev: 'Runnable' = None
        self.current: 'Runnable' = self

    def __or__(self, other) -> 'Runnable':
        other = coerse_to_runnable(other)
        
        self.next = other
        other.prev = self
        other.first = self.first

        return other
        
    def __ror__(self, other) -> 'Runnable':
        other = coerse_to_runnable(other)

        self.prev = other
        if other.first != other:
            self.first = other.first
        else:
            self.first = other
        
        other.next = self
        return self
    
    def __and__(self, other) -> 'Runnable':
        other = coerse_to_runnable(other)

        if isinstance(self, RunnableParallel):
            self.runnables.append(other)
            other.prev = self.prev
            other.first = self.first
            return self
        elif isinstance(other, RunnableParallel):
            other.runnables.append(self)
            self.prev = other.prev
            self.first = other.first
            return other
        else:
            other.prev = self.prev
            other.first = self.first
            return RunnableParallel(self, other)
        
    def __rand__(self, other) -> 'Runnable':
        other = coerse_to_runnable(other)

        if isinstance(self, RunnableParallel):
            self.runnables.append(other)
            other.prev = self.prev
            other.first = self.first
            return self
        elif isinstance(other, RunnableParallel):
            other.runnables.append(self)
            self.prev = other.prev
            self.first = other.first
            return other
        else:
            other.prev = self.prev
            other.first = self.first
            return RunnableParallel(self, other)

    @abstractmethod
    def _generate(self, **kwargs) -> Any:
        pass

    def generate(self, **kwargs) -> Any:
        current = self.first
        run_args = kwargs
        while current is not None:
            result = current._generate(**run_args)
            run_args.update(result)

            current = current.next
        
        return run_args

class RunnableParallel(Runnable):
    def __init__(self, *args: Runnable) -> None:
        super().__init__()
        self.runnables: list[Runnable] = args

    def generate(self, **kwargs) -> Any:
        results = []
        for runnable in self.runnables:
            result = runnable.generate(**kwargs)
            
            results.append(result)

        return results

class RunnableConst(Runnable):
    def __init__(self, value: any) -> None:
        super().__init__()
        self.value = value

    def _generate(self, **kwargs) -> any:
        return self.value

def coerse_to_runnable(value: any) -> Runnable:
    if isinstance(value, Runnable):
        return value
    return RunnableConst(value)