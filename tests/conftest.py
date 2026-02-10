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
from statek.executors.chat_log_item import ChatLogItem

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
        # Create empty db0 snapshot
        paths = {}
        paths["EMPTY_DB0"] = os.path.join(TEST_FILES_DIR_ROOT, "empty_db0")
        db0.commit()
        copy_directory(DB0_DIR, paths["EMPTY_DB0"])

        # Close db0
        db0.close()  # pylint: disable=no-member

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
def db0_fixture(db0_fixture_preloaded):
    """Provide empty db0 instance for each test."""
    with FixtureInitializationManager("db0_fixture"):
        if os.path.exists(DB0_DIR):
            shutil.rmtree(DB0_DIR)

        # Copy empty db0
        paths = db0_fixture_preloaded["db0_paths"]
        copy_directory(paths["EMPTY_DB0"], DB0_DIR)

        # Initialize db0
        db0.init(DB0_DIR, read_write=True)
        db0.open("test_prefix", "rw")  # pylint: disable=no-member
        yield db0
        db0.close()  # pylint: disable=no-member





@pytest.fixture
def agent(db0_fixture):  # pylint: disable=unused-argument
    """Create a test agent."""
    return Agent(role="test", _system_prompt="Test agent", _prompt_template="Test task", _tools=[])


@pytest.fixture
def agent_factory(db0_fixture):  # pylint: disable=unused-argument
    """Factory fixture to create Agent instances with custom parameters."""
    def _create_agent(prompt_template="Test task", role="test", system_prompt="Test", tools=None):
        return Agent(
            role=role,
            _system_prompt=system_prompt,
            _prompt_template=prompt_template,
            _tools=tools or []
        )
    return _create_agent


@pytest.fixture
def supervised_agent(db0_fixture):  # pylint: disable=unused-argument
    """Create a test agent."""
    return SupervisedAgent(
        role="test",
        _system_prompt="Test agent",
        _prompt_template="Test task",
        _tools=[]
    )


@pytest.fixture
def job_def_factory(agent):
    """Factory fixture to create JobDef instances with custom parameters."""
    def _create_job_def(job_params=None, warmup_code=None):
        return JobDef(
            agent=agent,
            job_params=job_params,
            warmup_code=warmup_code
        )
    return _create_job_def


@pytest.fixture
def job_factory(job_def_factory):
    """Factory fixture to create Job instances with custom parameters."""
    def _create_job(job_params=None, model_family="test", model="test-model"):
        job_def = job_def_factory(job_params=job_params)
        return Job(
            job_def=job_def,
            model_family=model_family,
            model=model,
            job_status=JobStatus.READY  # pylint: disable=no-member
        )
    return _create_job


def create_chat_log_item(console_pos, llm_resp):
    """Helper function to create ChatLogItem instances."""
    return ChatLogItem(
        console_pos=console_pos,
        llm_resp=llm_resp,
        timestamp=datetime.now()
    )


# Mock tool functions for testing
def clock():
    """Get the current time.

    Returns:
        str: The current timestamp.
    """
    return None


def docs(class_name: type, method_name: str = None) -> str:  # pylint: disable=unused-argument
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
        'docs': docs,
        'exit_tool': exit_tool
    }
