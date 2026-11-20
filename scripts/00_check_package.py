"""Validate that every required course input is readable and dimensionally valid."""

import argparse

from common import ROOT
from completion_inventory import (
    build_inventory,
    validate_cohort,
    verify_sha256_manifest,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--hashes',action='store_true')
    args=parser.parse_args()
    if not (ROOT/'data/cases.csv').is_file():
        parser.exit(1,'Data not installed. See DATA_DOWNLOAD.md: download BIOF3001_Data.zip and place its data folder alongside scripts.\n')
    inventory = build_inventory(ROOT)
    validate_cohort(inventory)
    if args.hashes:
        verify_sha256_manifest(ROOT, ROOT/'SHA256SUMS.txt')
    print(
        f"Checked {len(inventory['case_ids'])} cases, "
        f"{inventory['mesh_count']} individual meshes; CT volumes are not included."
    )


if __name__=='__main__': main()
