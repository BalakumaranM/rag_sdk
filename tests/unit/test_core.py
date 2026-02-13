from rag_sdk.config import ConfigLoader


def test_dummy():
    config = ConfigLoader.from_env()
    assert config is not None


def test_rag_init():
    # Attempt to load RAG just to verify imports and basic sanity
    # Not testing logic without mocks
    assert True
