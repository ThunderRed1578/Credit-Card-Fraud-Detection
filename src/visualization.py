import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_outlier_comparison(X_before, y_before, X_after, y_after, feat_indices, feat_names):
    n_feats = len(feat_indices)
    
    _, axes = plt.subplots(2, n_feats, figsize=(20, 10))
    
    # Trước khi lọc
    for i, (idx, name) in enumerate(zip(feat_indices, feat_names)):
        sns.boxplot(x=y_before, y=X_before[:, idx], ax=axes[0, i], palette="Set2", legend=False, hue=y_before)
        axes[0, i].set_title(f'{name} Before (Hàng gốc)')
        axes[0, i].set_xlabel('Class (0: Normal, 1: Fraud)')
        
    # Sau khi lọc
    for i, (idx, name) in enumerate(zip(feat_indices, feat_names)):
        sns.boxplot(x=y_after, y=X_after[:, idx], ax=axes[1, i], palette="Set2", legend=False, hue=y_after)
        axes[1, i].set_title(f'{name} After (Đã lọc)')
        axes[1, i].set_xlabel('Class (0: Normal, 1: Fraud)')

    plt.suptitle('Hiệu quả của việc lọc Outlier (Boxplot Comparison)', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_dist_before_filter(X_before, feat_indices, feat_names):
    n_feats = len(feat_indices)

    _, axes = plt.subplots(1, n_feats, figsize=(20, 4))
    for i, (idx, name) in enumerate(zip(feat_indices, feat_names)):
        sns.histplot(X_before[:, idx], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Dist of {name} (Before)')
    plt.tight_layout()
    plt.show()

def plot_model_report(results, model_name="Model"):
    """
    Vẽ báo cáo gồm 2 biểu đồ: Confusion Matrix và Precision-Recall Curve
    Input: results (dict) trả về từ hàm evaluate_comprehensive
    """
    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix
    cm = results['cm']
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
    
    # Tạo nhãn cho từng ô
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 12})
    axes[0].set_title(f'{model_name} Confusion Matrix', fontsize=15)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xticklabels(['Normal', 'Fraud'])
    axes[0].set_yticklabels(['Normal', 'Fraud'])
    
    # Precision-Recall Curve
    axes[1].plot(results['rec_curve'], results['prec_curve'], 
                 color='darkorange', lw=2, 
                 label=f'PR Curve (AUPRC = {results["auprc"]:.4f})')
    
    # Vẽ đường cơ sở
    baseline = results['baseline']
    axes[1].plot([0, 1], [baseline, baseline], linestyle='--', color='navy', label=f'No Skill ({baseline:.4f})')
    
    axes[1].set_title(f'{model_name} Precision-Recall Curve', fontsize=15)
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # In chỉ số tóm tắt dưới biểu đồ
    metrics_text = (
        f"Accuracy: {results['accuracy']:.4f} | "
        f"Precision: {results['precision']:.4f} | "
        f"Recall: {results['recall']:.4f} | "
        f"F1-Score: {results['f1']:.4f}"
    )
    
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.05, metrics_text, ha="center", fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":10})
    
    plt.show()

def plot_comparison_bar(results_dict):
    """
    Vẽ biểu đồ cột so sánh nhiều model.
    Input: results_dict = {'Model A': res_a, 'Model B': res_b}
    """
    metrics = ['AUPRC', 'F1', 'Recall', 'Precision']
    n_models = len(results_dict)
    
    data = {}
    for name, res in results_dict.items():
        data[name] = [res['auprc'], res['f1'], res['recall'], res['precision']]
        
    x = np.arange(len(metrics))
    width = 0.8 / n_models
    
    _, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, values) in enumerate(data.items()):
        rects = ax.bar(x + i*width, values, width, label=name)
        
        # Gán số lên cột
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3)
    
    plt.show()