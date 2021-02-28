"""Console script for johnstondechazal."""
import sys

import click

from johnstondechazal.data import PKG_DIR, download_data


@click.group()
def main():
    """Console script for johnstondechazal."""


@main.command()
@click.argument('dest', default=PKG_DIR)
def get_data(dest):
    """Download the Johnston & de Chazal dataset to DEST, where the default location is
    the johnstondechazal package directory"""
    download_data(dest)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
