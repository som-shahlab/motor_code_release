import collections.abc
import datetime
from typing import List, Optional

import numpy as np

import femr.extension

class Ontology:
    def __init__(self, *args, **kwargs) -> None: ...
    def get_all_parents(self, arg0: str) -> Sequence[str]: ...
    def get_children(self, arg0: str) -> Sequence[str]: ...
    def get_parents(self, arg0: str) -> Sequence[str]: ...

class PatientDatabase(collections.abc.Sequence):
    def __init__(self, filename: str, read_all: bool = ...) -> None: ...
    def close(self) -> None: ...
    def get_patient_birth_date(self, arg: int) -> datetime.datetime: ...
    def get_ontology(self) -> Ontology: ...
    def __getitem__(self, arg0: int) -> object: ...
    def __len__(self) -> int: ...

def convert_patient_collection_to_patient_database(arg0, arg1, arg2, arg3: str, arg4: int) -> None: ...
def sort_and_join_csvs(arg0, arg1, arg2: List[str] | np.dtype, arg3: str, arg4: int) -> None: ...