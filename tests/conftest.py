"""Conftest for statek test package."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import os
import shutil

import pytest
import dbzero as db0


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
        db0.open("test_prefix", "rw")
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
        db0.open("test_prefix", "rw")
        yield db0
        db0.close()  # pylint: disable=no-member
