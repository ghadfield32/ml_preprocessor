#!/usr/bin/env bash
QUARTO_VERSION=$1

echo "Installing Quarto version $QUARTO_VERSION"

# Identify the CPU type (M1 vs Intel)
# uname -m returns the machine hardware name.
# CPU variable is set based on the hardware type.
if [[ $(uname -m) ==  "aarch64" ]] ; then
  CPU="arm64"
elif [[ $(uname -m) ==  "arm64" ]] ; then
  CPU="arm64"
else
  CPU="amd64"
fi

# Download and install Quarto
# wget downloads the Quarto installation package.
# dpkg installs the downloaded package.
TEMP_QUARTO="$(mktemp)" && \
    wget -q  -O "$TEMP_QUARTO" https://github.com/quarto-dev/quarto-cli/releases/download/v$QUARTO_VERSION/quarto-${QUARTO_VERSION}-linux-${CPU}.deb && \
    dpkg -i "$TEMP_QUARTO" && \
    rm -f "$TEMP_QUARTO"