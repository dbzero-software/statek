"""Regression tests for dependency declarations."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_lines(path: str) -> list[str]:
    return [
        line.strip()
        for line in (REPO_ROOT / path).read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]


def test_ui_requirements_delegate_to_runtime_requirements():
    requirements = _read_lines('requirements-ui.txt')

    assert requirements == ['-r requirements.txt']


def test_runtime_requirements_install_local_modelkit_wheel():
    requirements = _read_lines('requirements.txt')

    assert './packages/dbzero_modelkit-0.1.0-py3-none-any.whl' in requirements


def test_pyproject_declares_modelkit_package_dependency():
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')

    assert '"dbzero-modelkit==0.1.0"' in pyproject
