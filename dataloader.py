import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 数据加载函数（复用DevNet实验的相同数据加载逻辑）
def data_loader(train_path, test_path):
    # 训练数据加载
    train_df = pd.read_csv(train_path)
    y_train = train_df['class'].values
    X_train = train_df.drop('class', axis=1).values

    # 测试数据加载
    test_df = pd.read_csv(test_path)
    y_test = test_df['class'].values
    X_test = test_df.drop('class', axis=1).values

    # 保持与DevNet相同的拆分比例（10%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    return X_train, y_train, X_test, y_test


# 可视化函数（复用DevNet实验的相同可视化逻辑）
def plot_results(y_true, scores, method_name):
    plt.figure(figsize=(15, 5))

    # 异常分数分布
    plt.subplot(131)
    plt.hist(scores[y_true == 0], bins=50, alpha=0.5, label='Normal')
    plt.hist(scores[y_true == 1], bins=50, alpha=0.5, label='Anomaly')
    plt.title(f'{method_name} Score Distribution')
    plt.legend()

    # ROC曲线
    plt.subplot(132)
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, scores):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # 准确率曲线
    plt.subplot(133)
    thresholds = np.linspace(np.min(scores), np.max(scores), 100)
    accuracies = [np.mean((scores > t) == y_true) for t in thresholds]
    best_thresh = thresholds[np.argmax(accuracies)]
    plt.plot(thresholds, accuracies)
    plt.axvline(best_thresh, color='r', linestyle='--',
                label=f'Best Threshold: {best_thresh:.2f}')
    plt.title('Accuracy vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 主执行流程
if __name__ == "__main__":
    # 数据加载
    X_train, y_train, X_test, y_test = data_loader("2test_data_snr20db.csv", "2test_data_snr20db.csv")

    # 异常检测算法配置
    models = {
        "Isolation Forest": IsolationForest(
            contamination=0.00,  # 与DevNet相同的异常比例
            random_state=42
        ),

        "OCSVM": OneClassSVM(
            nu=0.5,  # 异常比例
            kernel='rbf'  # 高斯核
        )
    }

    # 训练与评估
    results = {}
    for name, model in models.items():
        # 训练阶段
        if name != "LOF":  # LOF无监督不需要显式训练
            model.fit(X_train)

        # 预测异常分数
        if name == "LOF":
            scores = -model.decision_function(X_test)  # LOF分数需要取反
        else:
            scores = -model.decision_function(X_test) if hasattr(model, 'decision_function') \
                else model.score_samples(X_test)

        # 评估指标
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        results[name] = (roc_auc, pr_auc)

        # 可视化
        plot_results(y_test, scores, name)
        print(f"{name} Results:")
        print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}\n")

    # 结果对比
    print("\n=== Final Comparison ===")
    for name, (roc, pr) in results.items():
        print(f"{name}:")
        print(f"  ROC AUC: {roc:.4f}")
        print(f"  PR AUC: {pr:.4f}")
        print("-----------------------")