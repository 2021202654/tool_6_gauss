# graphene_tools.py (GPR 适配版)
import json
import numpy as np
import pandas as pd
import joblib # 替换 xgboost，改用 joblib 加载 sklearn 模型
from langchain.tools import tool
from graphene_features import enhance_features, calculate_theoretical_k

# === 全局配置 ===
# 注意后缀名变化
MODEL_PATH = "advanced_model.pkl" 
SCALER_PATH = "feature_scaler.pkl"
FEATURE_PATH = "model_features.json"

_gpr_model = None
_scaler = None
_model_features = None

def load_resources():
    """加载 GPR 模型、Scaler 和特征列表"""
    global _gpr_model, _scaler, _model_features
    
    if _model_features is None:
        try:
            with open(FEATURE_PATH, "r", encoding='utf-8') as f:
                _model_features = json.load(f)
        except Exception as e:
            return None, None, None, f"找不到特征文件: {str(e)}"
    
    if _scaler is None:
        try:
            _scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            return None, None, None, f"Scaler 加载失败: {str(e)}"

    if _gpr_model is None:
        try:
            _gpr_model = joblib.load(MODEL_PATH)
        except Exception as e:
            return None, None, None, f"模型加载失败: {str(e)}"
            
    return _gpr_model, _scaler, _model_features, ""

@tool
def ml_prediction_tool(
    length_um: float, 
    temperature_k: float, 
    defect_ratio: float, 
    layers: int = None,
    doping_ratio: float = None,
    is_suspended: bool = None,
    **kwargs 
) -> str:
    """[机器学习工具] 基于高斯过程回归 (GPR) 预测热导率及其不确定性。"""
    model, scaler, features, error_msg = load_resources()
    if error_msg: return f"系统错误: {error_msg}"

    try:
        defaults_log = []
        if layers is None: layers = 1; defaults_log.append("层数=1")
        if doping_ratio is None: doping_ratio = 0.0; defaults_log.append("掺杂=0.0%")
        if is_suspended is None: 
            is_suspended = True; current_substrate = 'Suspended'; defaults_log.append("基底=Suspended")
        else:
            current_substrate = 'Suspended' if is_suspended else 'SiO2'

        # 1. 构造原始数据
        raw_data = pd.DataFrame([{
            'length_um': length_um,
            'temperature': temperature_k,
            'defect_ratio': defect_ratio,
            'layers': layers,
            'doping_concentration': doping_ratio,
            'substrate_type': current_substrate
        }])
        
        # 2. 特征工程 & 空模具对齐 (Empty Mold)
        enhanced_input = enhance_features(raw_data)
        final_input = pd.DataFrame(0.0, index=[0], columns=features)
        
        for col in features:
            if col in enhanced_input.columns:
                final_input[col] = enhanced_input[col]
            elif col.startswith('substrate_type_'):
                target_type = col.replace('substrate_type_', '')
                if current_substrate == target_type:
                    final_input[col] = 1.0
        
        # 3. 🔥 标准化输入 (必须步骤)
        X_scaled = scaler.transform(final_input)
        
        # 4. 🔥 预测 (带标准差)
        # return_std=True 让 GPR 返回不确定性
        mean_log, std_log = model.predict(X_scaled, return_std=True)
        mean_log = mean_log[0]
        std_log = std_log[0] # 获取 log 空间下的标准差
        
        # 5. 还原数值与区间计算
        # 注意：Log空间下的加减，对应真实空间的乘除
        # 95% 置信区间 (2 sigma)
        pred_real = 10 ** mean_log - 1.0
        lower_bound = 10 ** (mean_log - 1.96 * std_log) - 1.0
        upper_bound = 10 ** (mean_log + 1.96 * std_log) - 1.0
        
        # 格式化输出：增加误差范围显示
        result_str = f"{pred_real:.2f} W/mK (95%置信区间: {lower_bound:.0f} ~ {upper_bound:.0f})"
        
        if defaults_log:
            note = ", ".join(defaults_log)
            return f"{result_str} | ℹ️ 自动补全: {note}"
        else:
            return result_str
        
    except Exception as e:
        return f"GPR 预测出错: {str(e)}"

@tool
def physics_calculation_tool(
    temperature_k: float, 
    defect_ratio: float, 
    length_um: float = 10.0, 
    **kwargs
) -> str:
    """[物理公式工具] 计算理论热导率上限，并返回物理机制拆解分析。"""
    try:
        temp_df = pd.DataFrame([{
            'temperature': temperature_k,
            'defect_ratio': defect_ratio,
            'length_um': length_um,
            'substrate_type': 'Suspended' 
        }])
        
        # 🔥 关键修改：获取物理组件详情
        k_val, components = calculate_theoretical_k(temp_df, return_components=True)
        k_final = k_val[0]
        
        # 格式化输出给 LLM 看
        # 我们把这些因子起个直观的名字，LLM 就能读懂了
        analysis_data = {
            "理论上限 (W/mK)": round(k_final, 2),
            "机制拆解": {
                "声子散射因子 (温度影响)": round(components['temp_factor'], 3),
                "边界散射因子 (尺寸影响)": round(components['size_factor'], 3),
                "点缺陷散射因子 (杂质影响)": round(components['defect_factor'], 3)
            }
        }
        
        return f"计算成功: {json.dumps(analysis_data, ensure_ascii=False)}"
        
    except Exception as e:
        return f"物理计算出错: {str(e)}"