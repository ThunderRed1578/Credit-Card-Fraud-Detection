import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15  # Tránh log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Khởi tạo zero hoặc random nhỏ
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            # z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias
            # y_pred = sigmoid(z)
            y_pred = self._sigmoid(linear_model)

            # dw = (1/m) * X.T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            # db = (1/m) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # w = w - learning_rate * dw
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and i % 100 == 0:
                loss = self._binary_cross_entropy(y, y_pred)
                self.losses.append(loss)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_pred_proba = self.predict_proba(X)
        class_preds = [1 if i > threshold else 0 for i in y_pred_proba]
        return np.array(class_preds)

def calculate_auprc_numpy(y_true, y_scores):
    """Hàm phụ trợ tính AUPRC"""
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    total_positives = np.sum(y_true)
    
    # Tránh chia cho 0
    precisions = np.divide(tps, (tps + fps), out=np.zeros_like(tps, dtype=float), where=(tps + fps)!=0)
    recalls = np.divide(tps, total_positives, out=np.zeros_like(tps, dtype=float), where=total_positives!=0)
    
    precisions = np.concatenate(([1], precisions))
    recalls = np.concatenate(([0], recalls))
    
    # Tính diện tích (AUPRC)
    if hasattr(np, 'trapezoid'):
        auprc = np.trapezoid(precisions, recalls)
    else:
        auprc = np.trapezoid(precisions, recalls)
        
    return auprc, precisions, recalls

def evaluate_comprehensive(y_true, y_pred_class, y_pred_proba):
    """
    Tính toán các chỉ số và trả về dictionary kết quả.
    Không in (print) gì cả.
    """
    # 1. Confusion Matrix
    TP = np.sum((y_true == 1) & (y_pred_class == 1))
    TN = np.sum((y_true == 0) & (y_pred_class == 0))
    FP = np.sum((y_true == 0) & (y_pred_class == 1))
    FN = np.sum((y_true == 1) & (y_pred_class == 0))
    
    cm = np.array([[TN, FP], [FN, TP]])
    
    # 2. Scalar Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 3. AUPRC Curve
    auprc, prec_curve, rec_curve = calculate_auprc_numpy(y_true, y_pred_proba)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "cm": cm,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "baseline": np.sum(y_true) / len(y_true)
    }