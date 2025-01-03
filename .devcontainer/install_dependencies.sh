#!/usr/bin/env bash

# Install system dependencies
# apt-get update updates the package list.
# apt-get install installs specified packages.
# Options:
#    Add or remove packages as needed for your environment.
apt-get update && apt-get install -y --no-install-recommends \
    python3-launchpadlib \
    vim \
    && rm -rf /var/lib/apt/lists/*