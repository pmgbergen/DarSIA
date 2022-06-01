# DarIA
Darcy scale image analysis toolbox


## Installation
Run the following to install:
```python
pip install daria
```

This might not be completely true (unless we upload to pypi). For now, I think that cloning from github and writing
```python
pip install daria
```
inside the directory is the correct approach.

## Usage

```python
We should add sample code here at some point
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

