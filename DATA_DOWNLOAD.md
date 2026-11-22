# Download the workshop data

The 130 MB course archive is stored in this GitHub repository as `data.zip`
using [Git Large File Storage](https://git-lfs.com/). Install Git LFS before
cloning so Git downloads the archive rather than only its small pointer file.

```bash
# macOS with Homebrew; use the Git LFS installer on Windows
brew install git-lfs
git lfs install
git clone https://github.com/grace-spec26/3001_DynamicSpine.git
cd 3001_DynamicSpine
```

If you already cloned the repository, install Git LFS and fetch the archive with:

```bash
git lfs install
git lfs pull
```

The expected SHA-256 checksum is:

```text
b3ddb3f6a6518b9839b32c65edb3fac13d04bb2844bebe298bb47fa67ce9efa2  data.zip
```

Verify it with `shasum -a 256 data.zip` on macOS/Linux or
`Get-FileHash data.zip -Algorithm SHA256` in PowerShell.

## Extract the data

`data.zip` contains four component archives and three CSV files. Extract the
outer archive into a new `data` folder, then extract each component archive into
that same folder.

On macOS/Linux:

```bash
mkdir -p data
unzip data.zip -d data
for archive in data/masks.zip data/meshes.zip data/merged_spines.zip data/transforms.zip; do
  unzip "$archive" -d data
done
```

In PowerShell:

```powershell
New-Item -ItemType Directory -Force data
Expand-Archive data.zip -DestinationPath data -Force
"masks", "meshes", "merged_spines", "transforms" | ForEach-Object {
    Expand-Archive "data/$_.zip" -DestinationPath data -Force
}
```

Keep the resulting `data` folder alongside `scripts`; do not put it inside
`scripts` or `outputs`.

The result should look like this:

```text
BIOF3001_Spine_Workshop/
    data/ 
        masks/
        meshes/
        merged_spines/
        transforms/
        cases.csv
        cohort_audit.csv
        label_map.csv
    scripts/
    instructions/
    templates/
    outputs/
    README.md
    requirements.txt
    SHA256SUMS.txt
```

Then follow the Python setup in `README.md` and run:

```powershell
python scripts/00_check_package.py --hashes
```

It should report **71 cases and 1,065 individual meshes**. The hash check verifies
that the downloaded files match the supplied dataset. Keep `DATA_SOURCES.md` for
the data attribution. The download includes segmentation masks, not CT images.

The extracted `data` directory is intentionally ignored by Git; `data.zip` is
the versioned source archive. Keep `DATA_SOURCES.md` with the project for data
attribution. The archive includes segmentation masks, not CT images, and does
not include completed answers or trained models.
