# Third-Party Notices

This project depends on third-party open-source packages. The table below
summarizes direct runtime dependencies and runtime transitive dependencies
identified from the current package metadata.

## Runtime Dependencies

| Package | Version reviewed | Relationship | License |
|---|---:|---|---|
| dbzero-modelkit | 1.1.1 | Direct runtime dependency | MIT |
| dbzero | 0.4.0 | Optional backend via dbzero extra; used directly by statek | LGPL-2.1 license text in package metadata |
| fasteners | 0.19 | Transitive via dbzero | Apache-2.0 |
| pydantic | 2.12.5 | Direct runtime dependency | MIT |
| pydantic-core | 2.41.5 | Transitive via pydantic | MIT |
| annotated-types | 0.7.0 | Transitive via pydantic | MIT |
| typing-extensions | 4.15.0 | Transitive via pydantic | PSF-2.0 |
| typing-inspection | 0.4.2 | Transitive via pydantic and pydantic-settings | MIT |
| pydantic-settings | 2.12.0 | Direct runtime dependency | MIT |
| python-dotenv | 1.2.2 | Transitive via pydantic-settings | BSD-3-Clause |
| httpx | 0.28.1 | Direct runtime dependency | BSD-3-Clause |
| httpcore | 1.0.9 | Transitive via httpx | BSD-3-Clause |
| anyio | 4.14.1 | Transitive via httpx | MIT |
| certifi | 2026.6.17 | Transitive via httpx | MPL-2.0 |
| idna | 3.18 | Transitive via httpx | BSD-3-Clause |
| nest-asyncio | 1.6.0 | Direct runtime dependency | BSD license in package metadata |
| RestrictedPython | 8.3 | Direct runtime dependency | ZPL-2.1 |

## Development Tools

These packages are used for testing, linting, building, or publishing and are
not runtime dependencies of the distributed package.

| Package | Version reviewed | Use | License |
|---|---:|---|---|
| pytest | 9.1.1 | Tests | MIT |
| pytest-asyncio | 1.4.0 | Async tests | Apache-2.0 |
| pylint | 4.0.6 | Linting | GPL-2.0-or-later |
| build | 1.5.0 | Package builds | MIT |
| twine | 6.2.0 metadata reviewed from PyPI | Publishing | Apache-2.0 |
