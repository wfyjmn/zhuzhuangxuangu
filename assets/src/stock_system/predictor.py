"""
预测生成模块
功能：基于XGBoost模型生成股票涨跌预测
"""
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockPredictor:
    """股票预测器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化预测器
        
        Args:
            config_path: 模型配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/model_config.json")
        
        self.config = self._load_config(config_path)
        self.model = None
        self.model_metadata = None
        self.threshold = self.config['xgboost']['threshold']
        self.features = self.config['data']['train_features']
        
        # 加载模型
        self._load_model()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载模型配置成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}")
            raise
    
    def _load_model(self):
        """加载XGBoost模型和元数据"""
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            
            model_path = os.path.join(workspace_path, self.config['xgboost']['model_path'])
            metadata_path = os.path.join(workspace_path, self.config['xgboost']['model_metadata_path'])
            
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 尝试加载模型
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"加载XGBoost模型成功: {model_path}")
            else:
                logger.warning(f"模型文件不存在，将创建新模型: {model_path}")
                self._create_dummy_model()
            
            # 尝试加载元数据
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"加载模型元数据成功: {metadata_path}")
            else:
                logger.warning(f"模型元数据文件不存在: {metadata_path}")
                self.model_metadata = {
                    'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'best_score': 0.0,
                    'params': self.config['xgboost']['params'],
                    'threshold': self.threshold,
                    'features': self.features
                }
        
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            # 创建一个简单的模型作为后备
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """创建一个简单的XGBoost模型作为后备"""
        logger.info("创建示例XGBoost模型...")
        
        # 创建简单的训练数据
        np.random.seed(42)
        n_samples = 1000
        n_features = len(self.features)
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # 设置参数
        params = self.config['xgboost']['params'].copy()
        
        # 训练模型
        self.model = xgb.train(params, dtrain, num_boost_round=100)
        
        logger.info("示例模型创建成功")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            特征DataFrame
        """
        try:
            # 确保所有特征都存在
            for feat in self.features:
                if feat not in data.columns:
                    # 如果特征不存在，用0填充
                    data[feat] = 0.0
            
            # 选择特征列
            feature_df = data[self.features].copy()
            
            # 填充缺失值
            feature_df = feature_df.fillna(0)
            
            # 确保数据类型正确
            for col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
            
            return feature_df
        except Exception as e:
            logger.error(f"准备特征数据失败: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对股票数据进行预测
        
        Args:
            data: 股票数据DataFrame，必须包含所有特征
            
        Returns:
            预测结果DataFrame，包含预测标签和概率
        """
        try:
            if self.model is None:
                logger.error("模型未加载")
                return pd.DataFrame()
            
            # 准备特征
            feature_df = self._prepare_features(data)
            
            # 转换为DMatrix
            dtest = xgb.DMatrix(feature_df.values)
            
            # 获取预测概率
            probabilities = self.model.predict(dtest)
            
            # 根据阈值确定预测标签
            predictions = (probabilities >= self.threshold).astype(int)
            
            # 构建结果DataFrame
            result_df = data.copy()
            result_df['predicted_label'] = predictions
            result_df['predicted_prob'] = probabilities
            result_df['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"预测完成，共 {len(result_df)} 条记录")
            logger.info(f"预测分布: 上涨={predictions.sum()}, 下跌={len(predictions)-predictions.sum()}")
            
            return result_df
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return pd.DataFrame()
    
    def predict_batch(self, stock_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        批量预测多只股票
        
        Args:
            stock_data_dict: 股票代码到特征数据的字典
            
        Returns:
            股票代码到预测结果的字典
        """
        result = {}
        
        for ts_code, data in stock_data_dict.items():
            try:
                # 添加股票代码列
                data = data.copy()
                data['ts_code'] = ts_code
                
                # 预测
                pred_result = self.predict(data)
                
                if not pred_result.empty:
                    result[ts_code] = pred_result
                
            except Exception as e:
                logger.error(f"预测股票 {ts_code} 失败: {e}")
                continue
        
        logger.info(f"批量预测完成，成功 {len(result)}/{len(stock_data_dict)} 只股票")
        return result
    
    def generate_features_from_price(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        从价格数据生成特征（示例实现）
        
        Args:
            price_data: 包含OHLCV的价格数据
            
        Returns:
            特征数据
        """
        try:
            df = price_data.copy()
            
            # 确保日期排序
            df = df.sort_values('trade_date')
            
            # 计算移动平均
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # 计算成交量相关指标
            df['volume_ma5'] = df['vol'].rolling(window=5).mean()
            df['volume_ratio'] = df['vol'] / df['volume_ma5']
            
            df['amount_ma5'] = df['amount'].rolling(window=5).mean()
            df['turnover_ratio'] = df['amount'] / df['amount_ma5']
            
            # 计算RSI (简化版)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算MACD (简化版)
            df['ema12'] = df['close'].ewm(span=12).mean()
            df['ema26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema12'] - df['ema26']
            
            # 计算KDJ (简化版)
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            df['kdj_k'] = rsv.rolling(window=3).mean()
            df['kdj_d'] = df['kdj_k'].rolling(window=3).mean()
            df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
            
            # 计算布林带
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            boll_std = df['close'].rolling(window=20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * boll_std
            df['boll_lower'] = df['boll_mid'] - 2 * boll_std
            
            # 计算涨跌幅
            df['price_change_5d'] = df['close'].pct_change(5)
            df['price_change_10d'] = df['close'].pct_change(10)
            df['volume_change_5d'] = df['vol'].pct_change(5)
            df['volume_change_10d'] = df['vol'].pct_change(10)
            
            # 计算振幅
            df['amplitude'] = (df['high'] - df['low']) / df['close'] * 100
            df['high_low_ratio'] = df['high'] / df['low']
            
            # 计算成交额占比和换手率（简化）
            df['amount_ratio'] = df['amount'] / df['amount'].rolling(window=20).mean()
            df['turnover_rate'] = df['vol'] / df['vol'].rolling(window=20).mean() * 100
            
            # 填充缺失值
            df = df.fillna(0)
            
            # 取最后一行作为当前特征
            features = df.iloc[-1:][self.features]
            
            logger.info(f"生成特征成功，共 {len(features[0])} 个特征")
            return features
        
        except Exception as e:
            logger.error(f"生成特征失败: {e}")
            return pd.DataFrame()
    
    def update_threshold(self, new_threshold: float):
        """
        更新预测阈值
        
        Args:
            new_threshold: 新的阈值
        """
        self.threshold = new_threshold
        logger.info(f"更新预测阈值为: {new_threshold}")
    
    def save_predictions(self, predictions: Dict[str, pd.DataFrame], filename: str = None):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果字典
            filename: 保存文件名
        """
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"predictions_{timestamp}.json"
            
            save_path = os.path.join(workspace_path, "assets/data", filename)
            
            # 转换为可序列化的格式
            save_data = {}
            for ts_code, df in predictions.items():
                save_data[ts_code] = df.to_dict('records')
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存预测结果成功: {save_path}")
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")


def test_predictor():
    """测试预测器"""
    predictor = StockPredictor()
    
    # 创建测试数据
    print("\n=== 测试预测功能 ===")
    np.random.seed(42)
    test_data = pd.DataFrame({
        'ts_code': ['600000.SH'],
        'trade_date': ['20241231']
    })
    
    # 生成特征
    for feat in predictor.features:
        test_data[feat] = np.random.randn()
    
    # 预测
    result = predictor.predict(test_data)
    print(f"预测结果:\n{result}")
    
    # 测试批量预测
    print("\n=== 测试批量预测 ===")
    stock_data = {}
    for i in range(5):
        data = pd.DataFrame({
            'ts_code': [f'60000{i}.SH'],
            'trade_date': ['20241231']
        })
        for feat in predictor.features:
            data[feat] = np.random.randn()
        stock_data[f'60000{i}.SH'] = data
    
    batch_result = predictor.predict_batch(stock_data)
    print(f"批量预测结果数量: {len(batch_result)}")


if __name__ == '__main__':
    test_predictor()
