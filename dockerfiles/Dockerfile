# This is the base Dockerfile for creating a DarSIA installation.
# The setup of the Docker image is under development and may change in the future.

# Base the Docker image on the official Python image.
# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Since using non-root users may lead to permission issues, we use root for now. See
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user for more.

ENV HOME /workdir
ENV DARSIA_HOME=${HOME}/darsia

# Create and move to code directory. This is where DarSIA will be installed.
WORKDIR ${HOME}

RUN apt-get update && \
    # Install missing packages.
    apt-get install -y git ffmpeg libsm6 libxext6 && \
    # Fetch DarSIA from GitHub to code directory
    git clone https://github.com/pmgbergen/DarSIA.git ${DARSIA_HOME}

# Move to DarSIA sub-directory
WORKDIR ${DARSIA_HOME}

# Update pip, install dependencies and freeze pip
RUN pip install --upgrade pip && \
    pip install -e.[dev] && \
    pip freeze && \
    # Remove git
    apt-get remove -y git python3-dev libssl-dev && \
    apt-get autoremove -y

# Add DarSIA home to the pythonpath. This may or may not be necessary.
ENV PYTHONPATH $DARSIA_HOME:$PYTHONPATH

# Run tests to check that everything works.
# NOTE: When run as a GH action (presumably after a PR to the develop branch
# was accepted, e.g., when the tests have already been run on GH), this should
# only fail if the docker image itself is flawed.
WORKDIR ${DARSIA_HOME}/tests
RUN pytest
WORKDIR ${DARSIA_HOME}