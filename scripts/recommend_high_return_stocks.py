#!/usr/bin/env python3
"""
高涨幅精准突击推荐系统
使用训练好的模型推荐5-10只高涨幅股票
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import akshare as ak

# 添加src到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.assault_features import AssaultFeatureEngineer


class HighReturnRecommender:
    """高涨幅精准突击推荐系统"""

    def __init__(self):
        """初始化推荐系统"""
        self.workspace_path = workspace_path
        self.model_dir = os.path.join(workspace_path, "assets/models")

        # 加载模型
        self.model = self._load_model()
        self.feature_names = self._load_features()
        self.metadata = self._load_metadata()

        # 初始化特征工程
        config_path = os.path.join(workspace_path, "config/short_term_assault_config.json")
        self.feature_engineer = AssaultFeatureEngineer(config_path)

        self.threshold = self.metadata.get('decision_threshold', 0.62)

    def _load_model(self):
        """加载训练好的模型"""
        model_path = os.path.join(self.model_dir, "high_return_100stocks_3years.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ 模型加载成功: {model_path}")
        return model

    def _load_features(self):
        """加载特征名称"""
        feature_path = os.path.join(self.model_dir, "high_return_100stocks_3years_features.pkl")
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ 特征加载成功: {len(feature_names)}个")
        return feature_names

    def _load_metadata(self):
        """加载元数据"""
        metadata_path = os.path.join(self.model_dir, "high_return_100stocks_3years_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✓ 元数据加载成功")
        print(f"  数据描述: {metadata['data_description']}")
        print(f"  训练日期: {metadata['training_date']}")
        print(f"  决策阈值: {metadata['decision_threshold']:.2f}")
        return metadata

    def get_stock_pool(self, n_stocks: int = 500) -> pd.DataFrame:
        """获取股票池"""
        print("\n" + "=" * 70)
        print("【步骤1】获取股票池")
        print("=" * 70)

        stock_list = ak.stock_info_a_code_name()
        stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停', na=False)]
        stock_list = stock_list.head(n_stocks)

        print(f"✓ 获取到 {len(stock_list)} 只股票")
        return stock_list

    def collect_latest_data(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """
        采集股票最新数据

        Args:
            stock_code: 股票代码
            days: 采集天数

        Returns:
            股票数据
        """
        try:
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

            # 获取历史行情
            df = ak.stock_zh_a_hist(symbol=stock_code,
                                   period="daily",
                                   start_date=start_date,
                                   end_date=end_date,
                                   adjust="qfq")

            df['stock_code'] = stock_code
            df['date'] = pd.to_datetime(df['日期'])
            df = df.drop(columns=['日期'])
            df = df.set_index('date')

            # 重命名列
            df = df.rename(columns={
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate'
            })

            return df

        except Exception as e:
            return None

    def predict_stock(self, stock_code: str, stock_name: str = "") -> Dict:
        """
        预测单只股票

        Args:
            stock_code: 股票代码
            stock_name: 股票名称

        Returns:
            预测结果
        """
        # 采集数据
        df = self.collect_latest_data(stock_code, days=100)
        if df is None or len(df) < 50:
            return None

        # 特征工程
        df = self.feature_engineer.create_all_features(df)
        if df.empty:
            return None

        # 获取最新一行的特征
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                        'amount', 'amplitude', 'pct_change', 'change_amount',
                        'turnover_rate', 'stock_code', 'date', 'returns', 'price_change',
                        'high_20', 'ema_12', 'ema_26', 'low_9', 'high_9',
                        'avg_volume_5', 'k_value', 'd_value', 'ma_5', 'ma_10', 'ma_20']

        available_features = [col for col in self.feature_names if col in df.columns]
        latest_features = df[available_features].iloc[-1:].values

        # 预测
        proba = self.model.predict_proba(latest_features)[0, 1]

        # 计算置信度（基于多个指标）
        confidence = self._calculate_confidence(df, proba)

        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'probability': proba,
            'confidence': confidence,
            'current_price': df['close'].iloc[-1],
            'change_pct': df['pct_change'].iloc[-1],
            'volume_ratio': df.get('volume_ratio', [0]).iloc[-1] if 'volume_ratio' in df.columns else 0,
            'rsi': df.get('enhanced_rsi', [50]).iloc[-1] if 'enhanced_rsi' in df.columns else 50
        }

    def _calculate_confidence(self, df: pd.DataFrame, proba: float) -> float:
        """
        计算综合置信度

        Args:
            df: 股票数据
            proba: 模型预测概率

        Returns:
            置信度 (0-100)
        """
        # 置信度 = 模型概率(60%) + 技术指标确认(40%)

        # 1. 模型概率（0-60分）
        model_score = min(proba * 100, 60)

        # 2. 技术指标确认（0-40分）
        tech_score = 0

        # RSI确认
        if 'enhanced_rsi' in df.columns:
            rsi = df['enhanced_rsi'].iloc[-1]
            if 50 < rsi < 80:  # RSI在合理区间
                tech_score += 10

        # 成交量确认
        if 'volume_ratio' in df.columns:
            vol_ratio = df['volume_ratio'].iloc[-1]
            if vol_ratio > 1.2:  # 成交量放大
                tech_score += 10

        # 价格趋势确认
        if 'momentum_5' in df.columns:
            momentum = df['momentum_5'].iloc[-1]
            if momentum > 0:  # 短期上涨趋势
                tech_score += 10

        # MACD确认
        if 'macd_golden_cross' in df.columns:
            macd_cross = df['macd_golden_cross'].iloc[-1]
            if macd_cross > 0:  # MACD金叉
                tech_score += 10

        confidence = model_score + tech_score
        return min(confidence, 100)

    def recommend(self, n_stocks: int = 500, top_n: int = 10) -> pd.DataFrame:
        """
        推荐高涨幅股票

        Args:
            n_stocks: 股票池大小
            top_n: 返回前N只股票

        Returns:
            推荐结果DataFrame
        """
        print("\n" + "=" * 70)
        print("高涨幅精准突击推荐")
        print("=" * 70)
        print(f"决策阈值: {self.threshold:.2f}")
        print(f"目标: 推荐前{top_n}只股票")
        print("=" * 70)

        # 获取股票池
        stock_pool = self.get_stock_pool(n_stocks)

        # 预测所有股票
        predictions = []
        for idx, row in stock_pool.iterrows():
            stock_code = row['code']
            stock_name = row['name']

            result = self.predict_stock(stock_code, stock_name)
            if result is not None and result['probability'] >= self.threshold:
                predictions.append(result)

            # 每50只股票打印一次进度
            if len(predictions) % 50 == 0:
                print(f"  已处理 {idx + 1}/{len(stock_pool)} 只股票")

        # 排序并返回top_n
        if not predictions:
            print("\n⚠ 没有找到符合条件的股票")
            return None

        # 按综合评分（置信度）排序
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        top_predictions = predictions[:top_n]

        # 转换为DataFrame
        df = pd.DataFrame(top_predictions)
        df = df.sort_values(by='confidence', ascending=False)

        # 打印推荐结果
        print("\n" + "=" * 70)
        print(f"✓ 找到 {len(predictions)} 只符合条件的股票")
        print(f"✓ 精选前 {top_n} 只推荐:")
        print("=" * 70)
        print(f"{'代码':<10} {'名称':<10} {'概率':<10} {'置信度':<10} {'涨幅':<10} {'量比':<10}")
        print("-" * 70)

        for idx, row in df.iterrows():
            print(f"{row['stock_code']:<10} {row['stock_name']:<10} "
                  f"{row['probability']:<10.2%} {row['confidence']:<10.1f} "
                  f"{row['change_pct']:<10.2f}% {row['volume_ratio']:<10.2f}")

        # 保存推荐结果
        report_dir = os.path.join(workspace_path, "assets/reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir,
                                  f"high_return_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 推荐结果已保存: {report_path}")
        print()

        return df


if __name__ == "__main__":
    recommender = HighReturnRecommender()
    results = recommender.recommend(n_stocks=500, top_n=10)
