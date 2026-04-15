"""Regression tests for dependency declarations."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_lines(path: str) -> list[str]:
    return [
        line.strip()
        for line in (REPO_ROOT / path).read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]


def test_runtime_requirements_include_web_ui_pdf_dependencies():
    requirements = _read_lines('requirements.txt')

    assert 'dbzero==0.1.12' in requirements
    assert 'nicegui==2.24.2' in requirements
    assert 'markdown2==2.5.4' in requirements
    assert 'weasyprint==68.1' in requirements


def test_ui_requirements_delegate_to_runtime_requirements():
    requirements = _read_lines('requirements-ui.txt')

    assert requirements == ['-r requirements.txt']
