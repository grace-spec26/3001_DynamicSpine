"""Provided split/CV/report plumbing; implement Exercise 3 to fit the regressor.

The fixed split evaluates regression within a SHARED SSM. It is not an external
validation of the whole shape-model pipeline. See the exercise instructions.
"""
import numpy as np
import joblib
from common import OUTPUT, manifest, read_centres, save_json
from regression_pipeline import fit_regression
from student_tasks import landmark_features


def main():
    pca=np.load(OUTPUT/'pca.npz',allow_pickle=False)
    ids=pca['case_ids']; y=pca['scores']
    split={r['case_id']:r['regression_split'] for r in manifest()}
    if any(c not in split for c in ids): raise ValueError('PCA case IDs must match the package manifest')
    x=landmark_features(np.stack([read_centres(c) for c in ids]))
    result=fit_regression(x,y,ids.tolist(),split)
    regression_dir=OUTPUT/'regression'; regression_dir.mkdir(parents=True,exist_ok=True)
    report=dict(cv=result['cv'],selected=result['selected'],
        test_rmse=result['test_rmse'],
        mean_score_baseline_rmse=result['mean_score_baseline_rmse'],
        training_cases=result['training_case_ids'],test_cases=result['test_case_ids'],
        scope='Regression holdout within a shared SSM; not external pipeline validation')
    save_json(regression_dir/'report.json',report)
    save_json(OUTPUT/'regression_report.json',report)
    np.savez(regression_dir/'predicted_scores.npz',case_ids=result['test_case_ids'],
        predicted=result['predicted'],actual=result['actual'])
    np.savez(OUTPUT/'predicted_scores.npz',case_ids=result['test_case_ids'],
        predicted=result['predicted'],actual=result['actual'])
    joblib.dump(result['model'],regression_dir/'pls_model.joblib')
    # Keep this estimator and its PCA basis together for the provided live viewer.
    # The viewer calls the same student feature/prediction/reconstruction functions.
    joblib.dump(dict(model=result['model'],mean=pca['mean'],modes=pca['modes'],
        eigenvalues=pca['eigenvalues'],
        case_ids=ids,centres=np.stack([read_centres(c) for c in ids]),
        default_case=str(result['test_case_ids'][0]),schema_version=2),OUTPUT/'interactive_model.joblib')
    print('Saved predictions and regression_report.json')
    print('Saved interactive_model.joblib; after Exercise 4, run scripts/07_interactive.py')
if __name__=='__main__': main()
