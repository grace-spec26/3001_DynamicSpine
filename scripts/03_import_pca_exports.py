"""Validate the standard PCA export layout and pack ``pca.npz``.

See instructions/03_shapeworks.md for the export layout and score convention.
No shape model is trained in this script. No student reconstruction is supplied.
"""
import argparse
from pathlib import Path

from common import OUTPUT
from import_pca_exports import pack_pca_exports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=str,default=str(OUTPUT/'pca_exports'))
    parser.add_argument('--modes',type=int,default=3)
    parser.add_argument('--output',type=Path,default=OUTPUT/'pca.npz')
    args=parser.parse_args()
    output = pack_pca_exports(Path(args.directory),args.output,args.modes)
    print('Saved',output,'with',args.modes,'modes')


if __name__=='__main__': main()
