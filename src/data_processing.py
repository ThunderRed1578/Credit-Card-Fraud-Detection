import numpy as np
import matplotlib.pyplot as plt

def numpy_pca(X, n_components=2):
    """
    Phân tích thành phần chính (PCA) sử dụng Eigendecomposition.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Tránh chia cho 0
    X_scaled = (X - X_mean) / X_std
    
    cov_matrix = np.cov(X_scaled, rowvar=False)
    
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    eigenvector_subset = sorted_eigenvectors[:, :n_components]
    
    X_reduced = np.dot(X_scaled, eigenvector_subset)
    
    return X_reduced

def numpy_truncated_svd(X, n_components=2):
    """
    Giảm chiều dữ liệu sử dụng SVD trực tiếp trên ma trận X.
    """
    X_centered = X - np.mean(X, axis=0)
    
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    U_reduced = U[:, :n_components]
    S_reduced = np.diag(S[:n_components])
    
    X_reduced = np.dot(U_reduced, S_reduced)
    
    return X_reduced

def numpy_tsne(X, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    """
    Cài đặt t-SNE giản lược (Exact t-SNE) bằng NumPy.
    """
    n_samples, n_features = X.shape
    
    # --- Helper: Tính khoảng cách Euclidean bình phương ---
    def calc_squared_distances(X):
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return D

    # --- Helper: Tính P-values (Gaussian Kernel ở chiều cao) ---
    def get_p_matrix(X, sigma=1.0):
        D = calc_squared_distances(X)
        P = np.exp(-D / (2 * sigma**2))
        np.fill_diagonal(P, 0.) # Không tính khoảng cách với chính nó
        P = P + 1e-12 # Tránh chia cho 0
        P = P / np.sum(P, axis=1, keepdims=True) # Chuẩn hóa theo hàng
        P = (P + P.T) / (2 * n_samples)
        return P

    print("t-SNE: Đang tính ma trận P...")
    P = get_p_matrix(X)
    P = P * 4.0 
    
    Y = np.random.randn(n_samples, n_components) * 1e-4
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    
    print(f"t-SNE: Bắt đầu {n_iter} vòng lặp...")
    for iter in range(n_iter):
        # q_ij = (1 + ||y_i - y_j||^2)^-1 / Sum(...)
        dist_Y = calc_squared_distances(Y)
        num = 1. / (1. + dist_Y)
        np.fill_diagonal(num, 0.)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # grad = 4 * Sum( (p_ij - q_ij) * num_ij * (y_i - y_j) )
        PQ = P - Q
        for i in range(n_samples):
            dY[i, :] = 4.0 * np.sum((PQ[i] * num[i])[:, np.newaxis] * (Y[i, :] - Y), axis=0)
            
        momentum = 0.5 if iter < 20 else 0.8
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < 0.01] = 0.01
        
        iY = momentum * iY - learning_rate * (gains * dY)
        Y = Y + iY
        Y = Y - np.mean(Y, 0) # Center lại map
        
        # Tắt Early exaggeration sau 100 vòng
        if iter == 100:
            P = P / 4.0
            print(f"  Vòng {iter}: Tắt early exaggeration")
            
        if iter % 100 == 0 and iter > 0:
            # Tính error (KL Divergence) để theo dõi
            error = np.sum(P * np.log(P / Q))
            print(f"  Vòng {iter}, Error: {error:.4f}")
            
    return Y

def numpy_smote(X, y, k_neighbors=5):
    """
    Thuật toán SMOTE (Synthetic Minority Over-sampling Technique).
    1. Lọc ra các điểm thuộc lớp thiểu số (Minority Class - Fraud).
    2. Với mỗi điểm, tìm k láng giềng gần nhất (KNN) bằng khoảng cách Euclidean.
    3. Chọn ngẫu nhiên một láng giềng và nội suy để tạo điểm mới.
    """
    # Tách lớp
    X_minority = X[y == 1]
    n_minority = len(X_minority)
    n_majority = np.sum(y == 0)
    
    # Số lượng cần sinh thêm để cân bằng 50/50
    n_synthetic = n_majority - n_minority
    
    if n_synthetic <= 0:
        return X, y
    
    synthetic_samples = []
    
    print(f"  - Bắt đầu sinh {n_synthetic} mẫu SMOTE từ {n_minority} mẫu gốc...")
    
    for i in range(n_synthetic):
        idx = np.random.randint(0, n_minority)
        sample = X_minority[idx]
        
        dists = np.sqrt(np.sum((X_minority - sample)**2, axis=1))
        
        neighbor_indices = np.argsort(dists)[1:k_neighbors+1]
        
        neighbor_idx = np.random.choice(neighbor_indices)
        neighbor = X_minority[neighbor_idx]
        
        gap = np.random.random()
        new_sample = sample + (neighbor - sample) * gap
        synthetic_samples.append(new_sample)
        
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.ones(len(X_synthetic))
    
    X_resampled = np.vstack((X, X_synthetic))
    y_resampled = np.concatenate((y, y_synthetic))
    
    return X_resampled, y_resampled

def remove_outliers_iqr(X, y, feature_indices, threshold=1.5):
    """
    Loại bỏ outliers sử dụng phương pháp IQR trên các features chỉ định.
    Áp dụng logic: Nếu giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR] thì loại bỏ.
    """
    mask = np.ones(len(X), dtype=bool)
    
    for idx in feature_indices:
        col_data = X[:, idx]
        q25, q75 = np.percentile(col_data, [25, 75])
        iqr = q75 - q25
        
        lower_bound = q25 - (threshold * iqr)
        upper_bound = q75 + (threshold * iqr)
        
        # Giữ lại các điểm nằm trong khoảng
        col_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
        mask = mask & col_mask
    
    print(f"  - Số lượng mẫu trước khi lọc: {len(X)}")
    print(f"  - Số lượng mẫu sau khi lọc: {np.sum(mask)}")
    print(f"  - Đã loại bỏ: {len(X) - np.sum(mask)} mẫu nhiễu.")
    return X[mask], y[mask]

def numpy_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def robust_scale_col(col):
    q25, q75 = np.percentile(col, [25, 75])
    iqr = q75 - q25
    if iqr == 0: iqr = 1
    return (col - np.median(col)) / iqr