"""Regression tests for dependency declarations."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_lines(path: str) -> list[str]:
    return [
        line.strip()
        for line in (REPO_ROOT / path).read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]


def test_ui_requirements_include_core_and_ui_runtime_requirements():
    requirements = _read_lines('requirements-ui.txt')

    assert requirements == [
        '-r requirements.txt',
        'nicegui==2.24.2',
        'markdown2==2.5.4',
        'weasyprint==68.1',
        'PyJWT==2.10.1',
    ]
