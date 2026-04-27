from abc import ABC, abstractmethod
from fastapi import FastAPI


class ApplicationCapability(ABC):
    @abstractmethod
    def enable(self, app: FastAPI):
        pass
