![build](https://github.com/EStorvik/DarIA/workflows/Build%20test/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: Apache v2](https://img.shields.io/hexpm/l/apa)](https://opensource.org/licenses/Apache-2.0)

# DarIA
Darcy scale image analysis toolbox

## Installation
Clone the repository from github and enter the DarIA folder. Then, run the following command to install:

```bash
pip install .
```

## Usage

```python
add some sample code here
```

## Developing DarIA
To install daria, along with the tools to develop and run tests, run the following in your virtualenv:
```bash
$ pip install -e .[dev]
```
-e means editable, and "[dev]" installs the development packages as well (currently only pytest).

At some point I will modify and make testing better, but for now:
<ul>
    <li> Write tests in the "tests"-folder</li>
    <ul>
        <li> File-names should start with "test_" </li>
        <li> Create methods whose name start with "test", and use assert </li>
    </ul>
    <li> run $pytest in the project directory.
<ul>

Use black (version 22.3.0) for formatting, this can be run as as a post-processing step, or directly in the IDE (e.g., bound to save.)

