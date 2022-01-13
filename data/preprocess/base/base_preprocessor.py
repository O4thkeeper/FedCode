from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, *args):
        pass