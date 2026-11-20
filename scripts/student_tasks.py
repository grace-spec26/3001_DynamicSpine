"""Implement these functions. Keep signatures and return shapes unchanged.

Run scripts/check_exercises.py after each exercise. TODOs intentionally raise
NotImplementedError; the surrounding file loading and saving is already supplied.
"""
from pathlib import Path
import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression

def label_centroid_ijk(mask, label):
    """Exercise 1A: mean voxel index of a nonzero vertebral label.

    mask: (I,J,K), as returned by nibabel; NOT SimpleITK's (K,J,I).
    Return: float array (3,) in i,j,k order. Raise ValueError if absent.
    """
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array")

    I, J, K = mask.shape
    sum_i = 0.0
    sum_j = 0.0
    sum_k = 0.0
    count = 0

    # 逐 i 切片处理，避免分配三个全尺寸坐标网格
    for i in range(I):
        slice_mask = mask[i, :, :]          # 形状 (J, K)
        j_idx, k_idx = np.where(slice_mask == label)
        n = j_idx.size
        if n > 0:
            sum_i += i * n                  # 该切片内所有体素的 i 坐标都是 i
            sum_j += j_idx.sum()
            sum_k += k_idx.sum()
            count += n

    if count == 0:
        raise ValueError(f"Label {label} not found in mask")

    return np.array([sum_i / count, sum_j / count, sum_k / count], dtype=float)

def apply_affine(points, matrix):
    """Exercise 1B: map (N,3) points using a (4,4) affine; return (N,3).

    Matrix convention: column vectors, p_out = M @ [x,y,z,1].
    Hint: append homogeneous ones and handle row/column orientation carefully.
    Do not perform another RAS/LPS flip: the supplied transform includes it.
    """
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if matrix.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")

    N = points.shape[0]

    # 添加齐次坐标：最后一列全为 1
    homog = np.hstack([points, np.ones((N, 1), dtype=float)])  # (N, 4)

    # 应用仿射：p_out = M @ homog.T，再转置回 (N, 4)
    transformed = (matrix @ homog.T).T  # (N, 4)

    # 取前三个坐标
    return transformed[:, :3]

def mesh_row(case_id, package_root, workbook_directory):
    """Exercise 2: return [case_id, '', path_T1, ..., path_L3] (17 strings).

    Use data/meshes/<case_id>/<level>.ply. Make paths relative to the OUTPUT
    workbook directory so moving the whole package preserves them. Validate
    every file exists. T1,T2,...,T12,L1,L2,L3 is anatomical, not lexical order.
    Hint: pathlib + os.path.relpath; forward slashes are portable.
    """
    # 解剖顺序：T1..T12, L1..L3
    levels = [f"T{i}" for i in range(1, 13)] + ["L1", "L2", "L3"]

    package_root = Path(package_root)
    workbook_directory = Path(workbook_directory)

    relative_paths = []
    for level in levels:
        abs_path = package_root / "data" / "meshes" / case_id / f"{level}.ply"
        if not abs_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {abs_path}")

        # 相对于输出工作簿目录的路径
        rel_path = os.path.relpath(abs_path, workbook_directory)
        # 统一使用正斜杠，保证跨平台可移植
        rel_path = Path(rel_path).as_posix()
        relative_paths.append(rel_path)

    # 返回 [case_id, '', path_T1, ..., path_L3]
    return [case_id, ""] + relative_paths

def landmark_features(centres):
    """Exercise 3A: (N,15,3) -> (N,45) features.

    Baseline: flatten the already aligned coordinates in anatomical order.
    Keep physical scale for this baseline. PLS will learn training-set scaling.
    Translation/chain-length normalization is an optional ablation, not required.
    Apply identical preprocessing at fit and prediction time.
    """
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 3 or centres.shape[1:] != (15, 3):
        raise ValueError("centres must have shape (N, 15, 3)")
    N = centres.shape[0]
    # 按 (15,3) 展平为 45 维，顺序为 T1_x, T1_y, T1_z, T2_x, ...
    return centres.reshape(N, 45)

def fit_pls(x_train, y_train, n_components):
    """Exercise 3B: return a fitted sklearn PLSRegression estimator.
    X: (N,45); Y: (N,K) raw PC scores. Use scale=True and max_iter=1000.
    Fit ONLY the training fold. Never fit a scaler on validation/test data.
    n_components counts PLS latent variables, NOT output PCA modes.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if x_train.ndim != 2 or x_train.shape[0] == 0 or x_train.shape[1] == 0:
        raise ValueError("x_train must be a nonempty 2D array")
    if y_train.ndim != 2 or y_train.shape[0] == 0 or y_train.shape[1] == 0:
        raise ValueError("y_train must be a nonempty 2D array")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have the same number of rows")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("training arrays must contain only finite values")

    # 确保 n_components 不超过特征数或样本数
    max_components = min(x_train.shape[0], x_train.shape[1])
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if n_components > max_components:
        raise ValueError(f"n_components={n_components} too large; max is {max_components}")

    model = PLSRegression(n_components=n_components, scale=True, max_iter=1000)
    model.fit(x_train, y_train)
    return model

def predict_scores(model, features):
    """Exercise 3C: use the fitted model; return (N,K) predictions."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is None:
        raise ValueError("model is not a fitted PLS estimator")
    if features.shape[1] != expected_features:
        raise ValueError(
            f"features must have {expected_features} feature columns, "
            f"got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values")

    preds = model.predict(features)
    # sklearn 的 PLSRegression.predict 返回 (N, K) 或 (N,)，确保是二维
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds

def reconstruct_particles(scores, mean, modes):
    """Exercise 4: (N,K), (3840,3), (K,11520) -> (N,3840,3).

    Each row of modes is one UNIT PCA eigenvector. Scores are RAW projection
    coefficients, not standard-deviation slider values. Combine the weighted
    modes with the flattened mean. Do not multiply by sqrt(eigenvalue) again.
    A zero-score vector must return the mean exactly.
    """
    scores = np.asarray(scores, dtype=float)
    mean = np.asarray(mean, dtype=float)
    modes = np.asarray(modes, dtype=float)

    if scores.ndim != 2:
        raise ValueError("scores must have shape (N, K)")
    if mean.ndim != 2 or mean.shape[1] != 3:
        raise ValueError("mean must have shape (P, 3)")
    if modes.ndim != 2:
        raise ValueError("modes must have shape (K, 3P)")
    if scores.shape[1] != modes.shape[0]:
        raise ValueError("scores columns must equal the number of modes")
    if modes.shape[1] != mean.size:
        raise ValueError("each mode must match the flattened mean size")
    if not all(np.isfinite(value).all() for value in (scores, mean, modes)):
        raise ValueError("scores, mean, and modes must contain only finite values")

    N = scores.shape[0]
    
    # 2. 将 mean (3840, 3) 展平为 (11520,)
    # 注意：使用 reshape(-1) 可以确保数据按照行优先 (point-major) 展平，
    # 这样刚好能和 modes 的 11520 维对应上。
    mean_flat = mean.reshape(-1)
    
    # 3. 计算加权模式
    # scores 形状是 (N, K)，modes 形状是 (K, 11520)
    # 矩阵乘法后得到 (N, 11520)
    weighted_modes = np.dot(scores, modes)
    
    # 4. 加上平均形状
    # 利用 numpy 的广播机制，将 (11520,) 加到 (N, 11520) 的每一行上
    reconstructed_flat = weighted_modes + mean_flat
    
    # 5. 重塑回三维形状 (N, P, 3)
    reconstructed = reconstructed_flat.reshape(N, mean.shape[0], 3)
    
    return reconstructed
