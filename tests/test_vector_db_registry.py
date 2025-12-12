# tests/test_vector_db_registry.py

from fitz_stack.vector_db import (
    register_vector_db_plugin,
    get_vector_db_plugin,
    VectorDBRegistryError,
)


class DummyDB:
    plugin_name = "dummy"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def upsert(self, *args, **kwargs):
        return True

    def search(self, *args, **kwargs):
        return []

    def delete_collection(self, *args, **kwargs):
        return True


def test_vector_db_registry_register_and_load():
    register_vector_db_plugin(DummyDB, plugin_name="dummy")

    cls = get_vector_db_plugin("dummy")
    assert cls is DummyDB

    inst = cls(a=1)
    assert isinstance(inst, DummyDB)
    assert inst.kwargs == {"a": 1}


def test_vector_db_unknown_raises():
    try:
        get_vector_db_plugin("idk")
        assert False, "Expected VectorDBRegistryError"
    except VectorDBRegistryError:
        pass
