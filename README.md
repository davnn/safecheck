[![Check Status](https://github.com/davnn/safecheck/actions/workflows/check.yml/badge.svg)](https://github.com/davnn/safecheck/actions?query=workflow%3Acheck)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/davnn/safecheck/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/davnn/safecheck/releases)
![Coverage Report](https://raw.githubusercontent.com/davnn/safecheck/main/assets/coverage.svg)

# safecheck

Opinionated combination of typechecking libraries. Safecheck is a (very) minimal wrapper of the following libraries to
provide a unified and simple-to-use interface:

- typechecking [beartype](https://github.com/beartype/)
- shapechecking [jaxtyping](https://github.com/google/jaxtyping)
- dispatch [plum](https://github.com/beartype/plum)
