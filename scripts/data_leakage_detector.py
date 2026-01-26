#!/usr/bin/env python3
"""
数据泄露检测工具
用于检测特征工程中是否存在未来数据泄露
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLeakageDetector:
    """数据泄露检测器"""

    def __init__(self):
        self.issues = []
        self.warnings = []

    def detect_future_columns(self, df: pd.DataFrame) -> List[str]:
        """
        检测DataFrame中是否包含未来数据相关的列名

        Args:
            df: 要检测的DataFrame

        Returns:
            检测到的未来数据列名列表
        """
        print("=" * 70)
        print("【检测1】未来数据列名检测")
        print("=" * 70)

        future_keywords = ['future', 'return_', 'target', 'shift_neg']
        future_cols = []

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in future_keywords):
                future_cols.append(col)

        if future_cols:
            print(f"⚠️ 发现 {len(future_cols)} 个未来数据列:")
            for col in future_cols:
                print(f"  - {col}")
            self.issues.extend(future_cols)
        else:
            print("✓ 未发现未来数据列名")

        print()
        return future_cols

    def check_temporal_validity(self, df: pd.DataFrame, date_col: str = None) -> Dict:
        """
        检查时间顺序是否正确

        Args:
            df: DataFrame
            date_col: 日期列名

        Returns:
            检测结果字典
        """
        print("=" * 70)
        print("【检测2】时间顺序验证")
        print("=" * 70)

        if date_col is None and 'date' in df.columns:
            date_col = 'date'
        elif date_col is None and df.index.name == 'date':
            date_col = df.index.name

        if date_col is None:
            print("⚠️ 未找到日期列，跳过时间顺序验证")
            return {}

        # 检查日期是否排序
        if date_col == df.index.name:
            dates = df.index
        else:
            dates = df[date_col]

        is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))

        if not is_sorted:
            print("❌ 日期未按时间顺序排列！")
            self.issues.append("date_not_sorted")
        else:
            print("✓ 日期已按时间顺序排列")

        print()
        return {"is_sorted": is_sorted}

    def check_feature_target_correlation(self, df: pd.DataFrame,
                                        target_col: str = 'label',
                                        feature_cols: List[str] = None,
                                        threshold: float = 0.95) -> List[str]:
        """
        检查特征与目标变量的相关性是否过高（可能存在数据泄露）

        Args:
            df: DataFrame
            target_col: 目标列名
            feature_cols: 特征列名列表
            threshold: 相关性阈值

        Returns:
            高相关性特征列表
        """
        print("=" * 70)
        print(f"【检测3】特征-目标相关性检测（阈值={threshold:.2f}）")
        print("=" * 70)

        if target_col not in df.columns:
            print(f"⚠️ 未找到目标列 {target_col}，跳过相关性检测")
            return []

        if feature_cols is None:
            # 自动选择数值型列作为特征
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        high_corr_features = []

        for col in feature_cols:
            if col in df.columns:
                corr = df[col].corr(df[target_col])
                if abs(corr) >= threshold:
                    high_corr_features.append((col, corr))
                    self.issues.append(f"high_corr_{col}")

        if high_corr_features:
            print(f"⚠️ 发现 {len(high_corr_features)} 个高相关性特征:")
            for col, corr in sorted(high_corr_features, key=lambda x: abs(x[1]), reverse=True):
                print(f"  - {col}: {corr:.4f}")
        else:
            print(f"✓ 未发现相关性 > {threshold:.2f} 的特征")

        print()
        return [item[0] for item in high_corr_features]

    def check_lookahead_bias(self, df: pd.DataFrame,
                             feature_cols: List[str] = None) -> List[str]:
        """
        检查是否存在前瞻性偏差（lookahead bias）

        Args:
            df: DataFrame
            feature_cols: 特征列名列表

        Returns:
            可能存在前瞻性偏差的特征列表
        """
        print("=" * 70)
        print("【检测4】前瞻性偏差检测")
        print("=" * 70)

        suspicious_patterns = [
            ('shift_neg', '负数shift可能使用未来数据'),
            ('future', '包含future关键字'),
            ('lead', '包含lead关键字'),
            ('tomorrow', '包含tomorrow关键字'),
            ('next_day', '包含next_day关键字')
        ]

        suspicious_features = []

        if feature_cols is None:
            feature_cols = df.columns.tolist()

        for col in feature_cols:
            for pattern, desc in suspicious_patterns:
                if pattern in col.lower():
                    suspicious_features.append((col, desc))
                    self.warnings.append(f"suspicious_{col}")

        if suspicious_features:
            print(f"⚠️ 发现 {len(suspicious_features)} 个可疑特征:")
            for col, desc in suspicious_features:
                print(f"  - {col}: {desc}")
        else:
            print("✓ 未发现明显的前瞻性偏差特征")

        print()
        return [item[0] for item in suspicious_features]

    def check_train_test_leakage(self, train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                key_cols: List[str] = ['stock_code', 'date']) -> Dict:
        """
        检查训练集和测试集之间是否存在数据泄露

        Args:
            train_df: 训练集
            test_df: 测试集
            key_cols: 用于判断重复的列

        Returns:
            检测结果
        """
        print("=" * 70)
        print("【检测5】训练集-测试集泄露检测")
        print("=" * 70)

        # 检查是否有完全相同的行
        train_set = set(tuple(row) for row in train_df[key_cols].values)
        test_set = set(tuple(row) for row in test_df[key_cols].values)

        overlap = train_set & test_set

        if overlap:
            print(f"❌ 发现 {len(overlap)} 个重复的样本！")
            self.issues.append("train_test_overlap")
            print("  这表明训练集和测试集存在数据泄露！")
        else:
            print("✓ 训练集和测试集没有重叠样本")

        # 检查时间范围是否重叠
        if 'date' in key_cols:
            train_dates = set(train_df['date'].values)
            test_dates = set(test_df['date'].values)

            date_overlap = train_dates & test_dates
            if date_overlap:
                print(f"⚠️ 训练集和测试集在日期上有 {len(date_overlap)} 天的重叠")
                self.warnings.append("date_overlap")
            else:
                print("✓ 训练集和测试集在时间上完全分离")

        print()
        return {"overlap_count": len(overlap)}

    def check_perfect_predictability(self, y_pred: np.ndarray,
                                    y_true: np.ndarray) -> Dict:
        """
        检查预测是否过于完美（可能存在数据泄露）

        Args:
            y_pred: 预测值
            y_true: 真实值

        Returns:
            检测结果
        """
        print("=" * 70)
        print("【检测6】完美预测检测（过拟合预警）")
        print("=" * 70)

        accuracy = (y_pred == y_true).mean()
        is_perfect = accuracy >= 0.99

        if is_perfect:
            print(f"❌ 预测准确率 {accuracy:.2%} 过于完美！")
            print("  这强烈暗示存在数据泄露或过拟合！")
            self.issues.append("perfect_prediction")
        else:
            print(f"✓ 预测准确率 {accuracy:.2%} 在合理范围内")

        print()
        return {"accuracy": accuracy, "is_perfect": is_perfect}

    def generate_report(self) -> str:
        """生成检测报告"""
        print("=" * 70)
        print("【检测报告】")
        print("=" * 70)

        report = []

        if not self.issues and not self.warnings:
            report.append("✓ 未发现明显的数据泄露问题")
            print("✓ 未发现明显的数据泄露问题")
        else:
            if self.issues:
                report.append(f"❌ 发现 {len(self.issues)} 个严重问题:")
                print(f"\n❌ 发现 {len(self.issues)} 个严重问题:")
                for i, issue in enumerate(self.issues, 1):
                    report.append(f"  {i}. {issue}")
                    print(f"  {i}. {issue}")

            if self.warnings:
                report.append(f"\n⚠️ 发现 {len(self.warnings)} 个警告:")
                print(f"\n⚠️ 发现 {len(self.warnings)} 个警告:")
                for i, warning in enumerate(self.warnings, 1):
                    report.append(f"  {i}. {warning}")
                    print(f"  {i}. {warning}")

            report.append("\n建议措施:")
            report.append("1. 仔细检查所有特征的计算逻辑")
            report.append("2. 确保特征只使用历史数据，不使用未来数据")
            report.append("3. 严格按时间顺序划分训练集和测试集")
            report.append("4. 完全移除所有未来数据列（如 future_return_*）")
            report.append("5. 使用滚动窗口交叉验证，避免信息泄露")

        print("\n" + "=" * 70)
        return "\n".join(report)

    def reset(self):
        """重置检测器状态"""
        self.issues = []
        self.warnings = []


def main():
    """示例用法"""
    # 创建示例数据
    dates = pd.date_range('2022-01-01', '2022-12-31')
    stock_codes = ['000001'] * len(dates)

    df = pd.DataFrame({
        'date': dates,
        'stock_code': stock_codes,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(10000, 100000, len(dates)),
        'label': np.random.randint(0, 2, len(dates))
    })

    # 故意添加未来数据
    df['future_return_5d'] = df['close'].pct_change(5).shift(-5)
    df['leakage_feature'] = df['label'] * 0.99  # 泄露特征

    # 使用检测器
    detector = DataLeakageDetector()

    # 执行各项检测
    detector.detect_future_columns(df)
    detector.check_temporal_validity(df, 'date')
    detector.check_feature_target_correlation(df, 'label', threshold=0.90)
    detector.check_lookahead_bias(df)

    # 模拟预测
    y_pred = (df['leakage_feature'] > 0.5).astype(int)
    y_true = df['label'].values
    detector.check_perfect_predictability(y_pred, y_true)

    # 生成报告
    report = detector.generate_report()

    print("\n检测结果示例完成")


if __name__ == "__main__":
    main()
