"""
FlexTime — pytest configuration
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow-running")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
