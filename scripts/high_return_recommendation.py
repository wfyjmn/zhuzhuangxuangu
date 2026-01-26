#!/usr/bin/env python3
"""
高涨幅精准推荐系统
核心目标：
1. 只推荐高概率大涨的股票（置信度>0.7）
2. 数量严格控制（5-10只）
3. 精准买入建议（精确率>70%）
4. 高回报预期（3-5天涨幅≥8%）
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加src到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.assault_features import AssaultFeatureEngineer
from stock_system.triple_confirmation import TripleConfirmation


class HighReturnRecommender:
    """高涨幅精准推荐器"""

    def __init__(self, 
                 model_path: str = None,
                 threshold: float = 0.7):
        """
        初始化推荐器
        
        Args:
            model_path: 模型路径
            threshold: 置信度阈值（默认0.7）
        """
        self.threshold = threshold
        self.model = None
        self.feature_names = None
        self.feature_engineer = None
        self.triple_confirmation = None
        self.metadata = None
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载模型"""
        model_dir = Path(model_path).parent
        
        # 加载模型
        self.model = pickle.load(open(model_path, 'rb'))
        
        # 加载特征名称
        feature_path = model_dir / "high_return_assault_features.pkl"
        if feature_path.exists():
            self.feature_names = pickle.load(open(feature_path, 'rb'))
        else:
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        # 加载元数据
        metadata_path = model_dir / "high_return_assault_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.threshold = self.metadata.get('optimal_threshold', self.threshold)
        
        # 初始化特征工程
        config_path = "config/short_term_assault_config.json"
        self.feature_engineer = AssaultFeatureEngineer(config_path)
        
        # 初始化三重确认
        self.triple_confirmation = TripleConfirmation(config_path)
        
        print(f"✓ 模型加载完成")
        print(f"  置信度阈值: {self.threshold:.2f}")
        print(f"  目标涨幅: {self.metadata.get('target_return', 0.08)*100:.0f}%")
        print(f"  预测周期: {self.metadata.get('prediction_horizon', [3,4,5])}天")

    def predict_stock(self, stock_data: pd.DataFrame) -> Dict:
        """
        预测单只股票的涨跌
        
        Args:
            stock_data: 股票数据（至少包含5天数据）
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        # 特征工程
        df_features = self.feature_engineer.create_all_features(stock_data)
        
        # 只使用实际存在的特征
        available_features = [f for f in self.feature_names if f in df_features.columns]
        
        missing_features = set(self.feature_names) - set(available_features)
        if missing_features:
            # 对于缺失的特征（通常是标签列），用0填充
            for feat in missing_features:
                df_features[feat] = 0
        
        # 提取特征
        X = df_features[self.feature_names].values[-1:]  # 使用最新数据
        
        # 预测概率
        proba = self.model.predict_proba(X)[0, 1]  # 正类概率
        
        # 三重确认（如果失败则返回默认值）
        try:
            confirmation_result = self.triple_confirmation.validate_all_confirmations(stock_data, len(stock_data) - 1)
            if confirmation_result is None:
                # 三重确认失败，使用默认值
                confirmation_result = {
                    'capital': 0.5,
                    'sentiment': 0.5,
                    'technical': 0.5
                }
            else:
                # 将结果转换为统一格式
                confirmation_result = {
                    'capital': confirmation_result.get('capital', {}).get('score', 0.5) / 100 if confirmation_result.get('capital', {}).get('score', 0) > 0 else 0.5,
                    'sentiment': confirmation_result.get('sentiment', {}).get('score', 0.5) / 100 if confirmation_result.get('sentiment', {}).get('score', 0) > 0 else 0.5,
                    'technical': confirmation_result.get('technical', {}).get('score', 0.5) / 100 if confirmation_result.get('technical', {}).get('score', 0) > 0 else 0.5
                }
        except Exception as e:
            print(f"三重确认失败: {e}")
            confirmation_result = {
                'capital': 0.5,
                'sentiment': 0.5,
                'technical': 0.5
            }
        
        # 综合评分
        # 置信度占60%，三重确认占40%
        confirmation_score = (confirmation_result['capital'] + 
                            confirmation_result['sentiment'] + 
                            confirmation_result['technical']) / 3
        final_score = 0.6 * proba + 0.4 * confirmation_score
        
        return {
            'symbol': stock_data['symbol'].iloc[-1] if 'symbol' in stock_data.columns else '未知',
            'date': stock_data.index[-1].strftime('%Y-%m-%d'),
            'confidence': proba,
            'confirmation_score': confirmation_score,
            'final_score': final_score,
            'capital_confirmation': confirmation_result['capital'],
            'sentiment_confirmation': confirmation_result['sentiment'],
            'technical_confirmation': confirmation_result['technical'],
            'is_recommended': final_score >= self.threshold
        }

    def recommend_stocks(self, 
                       stock_pool: List[pd.DataFrame],
                       max_recommendations: int = 10) -> List[Dict]:
        """
        从股票池中推荐高涨幅股票
        
        Args:
            stock_pool: 股票池（每只股票的数据）
            max_recommendations: 最大推荐数量（5-10只）
        
        Returns:
            推荐列表（按综合评分排序）
        """
        print("\n" + "=" * 70)
        print("高涨幅精准推荐系统")
        print("=" * 70)
        print(f"股票池大小: {len(stock_pool)}只")
        print(f"置信度阈值: {self.threshold:.2f}")
        print(f"最大推荐数量: {max_recommendations}只")
        print("=" * 70)
        
        predictions = []
        
        # 对每只股票进行预测
        for i, stock_data in enumerate(stock_pool, 1):
            try:
                pred = self.predict_stock(stock_data)
                predictions.append(pred)
                print(f"[{i}/{len(stock_pool)}] {pred['symbol']}: 置信度={pred['confidence']:.2f}, 综合分={pred['final_score']:.2f}")
            except Exception as e:
                print(f"[{i}/{len(stock_pool)}] 预测失败: {e}")
                continue
        
        # 筛选推荐股票
        recommended = [p for p in predictions if p['is_recommended']]
        
        # 按综合评分排序
        recommended.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 限制数量
        recommended = recommended[:max_recommendations]
        
        print("\n" + "=" * 70)
        print(f"推荐结果: {len(recommended)}只股票")
        print("=" * 70)
        
        return recommended

    def format_recommendations(self, recommendations: List[Dict]) -> str:
        """格式化推荐结果"""
        if not recommendations:
            return "无推荐股票"
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("高涨幅精准推荐清单")
        report.append("=" * 70)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"推荐数量: {len(recommendations)}只")
        report.append(f"目标涨幅: ≥{self.metadata.get('target_return', 0.08)*100:.0f}%")
        report.append(f"预测周期: {self.metadata.get('prediction_horizon', [3,4,5])}天")
        report.append("=" * 70)
        
        # 表头
        report.append("\n| 排名 | 股票代码 | 日期 | 置信度 | 确认得分 | 综合评分 | 资金 | 情绪 | 技术 |")
        report.append("|------|----------|------|--------|----------|----------|------|------|------|")
        
        # 数据
        for i, rec in enumerate(recommendations, 1):
            report.append(
                f"| {i} | {rec['symbol']} | {rec['date']} | "
                f"{rec['confidence']:.2%} | {rec['confirmation_score']:.2f} | "
                f"{rec['final_score']:.2f} | "
                f"{'✓' if rec['capital_confirmation'] > 0.5 else '✗'} | "
                f"{'✓' if rec['sentiment_confirmation'] > 0.5 else '✗'} | "
                f"{'✓' if rec['technical_confirmation'] > 0.5 else '✗'} |"
            )
        
        # 详细说明
        report.append("\n" + "=" * 70)
        report.append("详细分析")
        report.append("=" * 70)
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"\n【{i}. {rec['symbol']}】")
            report.append(f"  日期: {rec['date']}")
            report.append(f"  模型置信度: {rec['confidence']:.2%}")
            report.append(f"  三重确认得分: {rec['confirmation_score']:.2f}")
            report.append(f"  综合评分: {rec['final_score']:.2f}")
            report.append(f"  资金确认: {rec['capital_confirmation']:.2f}")
            report.append(f"  情绪确认: {rec['sentiment_confirmation']:.2f}")
            report.append(f"  技术确认: {rec['technical_confirmation']:.2f}")
        
        # 操作建议
        report.append("\n" + "=" * 70)
        report.append("操作建议")
        report.append("=" * 70)
        report.append("  1. 严格按照推荐顺序建仓（综合评分从高到低）")
        report.append("  2. 单只股票仓位建议：总资金的10-15%")
        report.append("  3. 止损位：买入价 -5%")
        report.append("  4. 止盈目标：+8%（3-5天内）")
        report.append("  5. 如3天内未达到+8%，考虑平仓")
        report.append("  6. 严格执行止损，不要犹豫")
        report.append("=" * 70)
        
        return "\n".join(report)

    def save_recommendations(self, 
                          recommendations: List[Dict],
                          output_dir: str = None):
        """保存推荐结果"""
        if output_dir is None:
            output_dir = os.path.join(workspace_path, "assets/recommendations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON
        json_path = os.path.join(output_dir, f"high_return_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON已保存: {json_path}")
        
        # 保存CSV
        df = pd.DataFrame(recommendations)
        csv_path = os.path.join(output_dir, f"high_return_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV已保存: {csv_path}")
        
        # 保存报告
        report_path = os.path.join(output_dir, f"high_return_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.format_recommendations(recommendations))
        print(f"✓ 报告已保存: {report_path}")
        
        return {
            'json_path': json_path,
            'csv_path': csv_path,
            'report_path': report_path
        }


def generate_mock_stock_pool(n_stocks: int = 100, 
                            n_days: int = 60) -> List[pd.DataFrame]:
    """生成模拟股票池"""
    stock_pool = []
    
    np.random.seed(42)
    
    for i in range(n_stocks):
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # 随机生成股票数据
        returns = np.random.normal(0.003, 0.025, n_days)
        prices = 10 + np.random.rand() * 90  # 10-100元
        prices = prices * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': np.roll(prices, 1) * (1 + np.random.normal(0, 0.015, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            'volume': np.random.lognormal(11, 0.6, n_days),
            'main_net_inflow': np.random.normal(0, 0.5, n_days),
            'big_buy_amount': np.random.exponential(50, n_days)
        })
        
        df.loc[0, 'open'] = df.loc[0, 'close']
        df['total_amount'] = df['close'] * df['volume'] / 10000  # 添加成交额列
        df['symbol'] = f"{i+1:06d}"  # 6位代码
        df.set_index('date', inplace=True)
        
        stock_pool.append(df)
    
    print(f"✓ 生成了{n_stocks}只股票的模拟数据")
    return stock_pool


def main():
    """主函数"""
    print("=" * 70)
    print("高涨幅精准推荐系统")
    print("=" * 70)
    
    # 1. 加载模型
    print("\n【步骤1】加载模型")
    print("-" * 70)
    model_path = os.path.join(workspace_path, "models/high_return_assault_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        print("请先运行: python scripts/train_high_return_assault.py")
        return
    
    recommender = HighReturnRecommender(model_path=model_path)
    
    # 2. 生成模拟股票池
    print("\n【步骤2】生成股票池")
    print("-" * 70)
    stock_pool = generate_mock_stock_pool(n_stocks=100, n_days=60)
    
    # 3. 推荐股票
    print("\n【步骤3】生成推荐")
    print("-" * 70)
    recommendations = recommender.recommend_stocks(stock_pool, max_recommendations=10)
    
    # 4. 输出推荐
    print("\n" + "=" * 70)
    print("推荐结果")
    print("=" * 70)
    print(recommender.format_recommendations(recommendations))
    
    # 5. 保存推荐
    print("\n【步骤4】保存推荐")
    print("-" * 70)
    paths = recommender.save_recommendations(recommendations)
    
    print("\n" + "=" * 70)
    print("✨ 推荐完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
