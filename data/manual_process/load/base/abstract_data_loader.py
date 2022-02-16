from abc import ABC, abstractmethod

# todo delete file
class BaseRawDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path):
        self.data_path = data_path
        self.attributes = dict()
        self.attributes["index_list"] = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_data_file(self, file_path):
        pass

    @abstractmethod
    def generate_h5_file(self, file_path):
        pass