"""Exercise 4: reconstruct test shapes and export particles by vertebral level."""
import numpy as np
import shutil
from common import OUTPUT, LEVELS
from reconstruction_pipeline import build_reconstructions, write_reconstructions


def main():
    pca=np.load(OUTPUT/'pca.npz',allow_pickle=False)
    predictions=np.load(OUTPUT/'predicted_scores.npz',allow_pickle=False)
    result=build_reconstructions(predictions['case_ids'].tolist(),predictions['predicted'],
        predictions['actual'],pca['mean'],pca['modes'],list(LEVELS),256)
    reconstruction_dir=OUTPUT/'reconstructions'
    write_reconstructions(reconstruction_dir,result,list(LEVELS),256)
    shutil.copy2(reconstruction_dir/'reconstructions.npz',OUTPUT/'reconstructions.npz')
    shutil.copy2(reconstruction_dir/'particle_errors.json',OUTPUT/'particle_errors.json')
    compatibility=OUTPUT/'particles'
    if compatibility.exists():
        shutil.rmtree(compatibility)
    shutil.copytree(reconstruction_dir/'particles',compatibility)
    print('Saved particles; errors compare predicted vs actual-score reconstruction, not ground-truth surface')


if __name__=='__main__': main()
