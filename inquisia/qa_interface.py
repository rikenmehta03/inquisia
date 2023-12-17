from abc import ABC, abstractmethod

class QAInterface(ABC):
    """
    Abstract class for Query/Answering(QA) interface.
    Can be used to implement custom QA generation system.
    """

    @abstractmethod
    def generate_answer(self, question: str) -> str:
        raise NotImplementedError("generate_answer is an abstract method and must be implemented")