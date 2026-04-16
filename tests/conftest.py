"""Conftest for statek test package."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from datetime import datetime
import os
import shutil

import pytest
import dbzero as db0

from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent, SupervisedAgent
from statek.executors.chat_log_item import LLM_LogItem, WarmupLogItem

TEST_FILES_DIR_ROOT = os.path.join(os.getcwd(), "__test_files")
TEST_DIR = os.path.join(os.path.dirname(__file__))
DB0_DIR = os.path.join(TEST_FILES_DIR_ROOT, "db0")


FIXTURE_INITIALIZED = None


class FixtureInitializationManager:
    """Manager to ensure only one fixture initializes at a time."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global FIXTURE_INITIALIZED  # pylint: disable=global-statement
        if FIXTURE_INITIALIZED is not None:
            raise RuntimeError(
                f"Trying to initialize fixture {self.name} but already "
                f"initialized by {FIXTURE_INITIALIZED}."
            )
        FIXTURE_INITIALIZED = self.name

    def __exit__(self, exc_type, exc_value, traceback):
        global FIXTURE_INITIALIZED  # pylint: disable=global-statement
        FIXTURE_INITIALIZED = None


def copy_directory(input_path, output_path):
    """Copy directory from input_path to output_path."""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(input_path, output_path)


def remove_lock_files(root_path):
    """Remove stale db0 lock files left behind after close()."""
    if not os.path.exists(root_path):
        return
    for current_root, _dirs, files in os.walk(root_path):
        for name in files:
            if name.endswith('.lock'):
                os.remove(os.path.join(current_root, name))


@pytest.fixture(scope='session')
def test_data_dir():
    """Provide test data directory path."""
    if not os.path.exists(TEST_FILES_DIR_ROOT):
        os.makedirs(TEST_FILES_DIR_ROOT)

    yield TEST_FILES_DIR_ROOT

    # Cleanup after all tests
    if os.path.exists(TEST_FILES_DIR_ROOT):
        shutil.rmtree(TEST_FILES_DIR_ROOT)


@pytest.fixture()
def temp_dir(test_data_dir):
    """Provide a temporary directory for test files."""
    temp_path = os.path.join(test_data_dir, "temp")

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    yield temp_path

    # Cleanup after test
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture(scope='session')
def db0_fixture_preloaded():
    """Create and initialize db0 database for tests."""
    with FixtureInitializationManager("db0_fixture_preloaded"):
        # Create db0 directory
        if os.path.exists(DB0_DIR):
            shutil.rmtree(DB0_DIR)
        os.makedirs(DB0_DIR)

        # Initialize db0
        db0.init(DB0_DIR, read_write=True)
        db0.open("test_prefix", "rw")  # pylint: disable=no-member
        db0.commit()
        db0.close()  # pylint: disable=no-member
        remove_lock_files(DB0_DIR)
        # Create empty db0 snapshot only after close so lock files are not copied
        paths = {}
        paths["EMPTY_DB0"] = os.path.join(TEST_FILES_DIR_ROOT, "empty_db0")
        copy_directory(DB0_DIR, paths["EMPTY_DB0"])

    yield {
        "db0_paths": paths,
    }

    # Cleanup after all tests
    for path in paths.values():
        if os.path.exists(path):
            shutil.rmtree(path)
    if os.path.exists(TEST_FILES_DIR_ROOT):
        shutil.rmtree(TEST_FILES_DIR_ROOT)


@pytest.fixture()
def db0_fixture():
    """Provide empty db0 instance for each test."""
    with FixtureInitializationManager("db0_fixture"):
        if os.path.exists(DB0_DIR):
            shutil.rmtree(DB0_DIR)
        os.makedirs(DB0_DIR)

        db0.init(DB0_DIR, read_write=True)
        db0.open("test_prefix", "rw")  # pylint: disable=no-member
        yield db0
        db0.close()  # pylint: disable=no-member





@pytest.fixture
def agent(db0_fixture):  # pylint: disable=unused-argument
    """Create a test agent."""
    return Agent(role="test", _system_prompt="Test agent", _tools=[],
                 _metadata={'MODEL': 'test-model'} )


@pytest.fixture
def agent_factory(db0_fixture):  # pylint: disable=unused-argument
    """Factory fixture to create Agent instances with custom parameters."""
    def _create_agent(role="test", system_prompt="Test", tools=None):
        return Agent(
            role=role,
            _system_prompt=system_prompt,
            _metadata={'MODEL': 'test-model'},
            _tools=tools or []
        )
    return _create_agent


@pytest.fixture
def supervised_agent(db0_fixture):  # pylint: disable=unused-argument
    """Create a test agent."""
    return SupervisedAgent(
        role="test",
        _system_prompt="Test agent",
        _metadata={'MODEL': 'test-model'},
        _tools=[]
    )


@pytest.fixture
def job_def_factory(agent):
    """Factory fixture to create JobDef instances with custom parameters."""
    def _create_job_def(
        job_params=None,
        warmup_code=None,
        model_family="test",
        model="test-model",
        **kwargs,
    ):
        return JobDef(
            agent=agent,
            job_params=job_params,
            warmup_code=warmup_code,
            model_family=model_family,
            model=model,
            **kwargs
        )
    return _create_job_def


@pytest.fixture
def job_factory(job_def_factory):
    """Factory fixture to create Job instances with custom parameters."""
    def _create_job(job_params=None, warmup_code=None, model_family="test", model="test-model"):
        job_def = job_def_factory(
            job_params=job_params,
            warmup_code=warmup_code,
            model_family=model_family,
            model=model,
        )
        return Job(
            job_def=job_def,
            job_status=JobStatus.READY  # pylint: disable=no-member
        )
    return _create_job


def create_chat_log_item(console_pos, llm_resp):
    """Helper function to create LLM_LogItem instances."""
    return LLM_LogItem(
        console_pos=console_pos,
        llm_resp=llm_resp,
        timestamp=datetime.now()
    )


def set_warmup_positions(job, positions):
    """Add WarmupLogItems to job.chat_log to simulate warmup_console_positions.

    Each position becomes a WarmupLogItem with console_pos derived from the
    previous position (0 for the first).  The end position is derived at
    read-time from the next element's console_pos in chat_log.
    Inserts at the front of chat_log to maintain correct ordering.
    """
    warmup_items = db0.list()  # pylint: disable=no-member
    prev = 0
    for i, end_pos in enumerate(positions):
        warmup_items.append(WarmupLogItem(
            console_pos=prev,
            warmup_block_num=i,
        ))
        prev = end_pos
    job.chat_log = warmup_items + job.chat_log


# Mock tool functions for testing
def clock():
    """Get the current time.

    Returns:
        str: The current timestamp.
    """
    return None


def docstr(class_name: type, method_name: str = None) -> str:  # pylint: disable=unused-argument
    """Print documentation for a class or method.

    Args:
        class_name (type): The class to document.
        method_name (str): Optional method name.

    Returns:
        str: The documentation string.
    """
    return ""


def exit_tool(reason: str) -> None:  # pylint: disable=unused-argument
    """Exit the current session.

    Args:
        reason (str): The reason for exiting.

    Returns:
        None: No return value.
    """
    return None


@pytest.fixture
def mock_tools():
    """Provide common mock tools for testing."""
    return {
        'clock': clock,
        'docstr': docstr,
        'exit_tool': exit_tool
    }
