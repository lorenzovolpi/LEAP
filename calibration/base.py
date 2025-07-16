from abc import ABC, abstractmethod


class SourceTargetCalibratorFactory(ABC):
    @abstractmethod
    def __call__(self, Zsrc, ysrc, Ztgt): ...
