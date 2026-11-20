"""Create demo or full ShapeWorks workbooks from the validated cohort."""

import argparse

from common import OUTPUT, ROOT
from create_project import create_project


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset',choices=['demo','all'],default='all')
    parser.add_argument(
        '--unique-mesh-names',
        action='store_true',
        help='Stage meshes with case-qualified names to prevent particle collisions',
    )
    parser.add_argument(
        '--output-name',
        help='Explicit workbook filename inside outputs, for example full_run.xlsx',
    )
    args=parser.parse_args()
    path = create_project(
        package_root=ROOT,
        output_dir=OUTPUT,
        template_path=ROOT/'templates/shapeworks_example.xlsx',
        subset=args.subset,
        unique_mesh_names=args.unique_mesh_names,
        output_name=args.output_name,
    )
    print(path)


if __name__=='__main__': main()
