import pandas as pd
import numpy as np
from pathlib import Path

"""
sklearn 是 Python 最常用的经典机器学习库。它负责把“数学上的机器学习算法”做成便于工程调用的接口。
里面装着各种工具：
- 分类器（SVM、Logistic）
- 回归器（Linear、Ridge、Lasso）
- 特征处理器（OneHotEncoder、StandardScaler）
- 评估方式（MAE、RMSE）
- 交叉验证（KFold）
- Pipeline（把多步组合起来）
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer

# CONFIG
DATA_PATH = "./data/ecommerce_sales_dataset.csv"  # 数据文件路径

"""
LinearRegression —— 普通线性回归：
找一组参数w和偏置b,让预测值尽可能接近真实y。没有限制参数大小、数据噪声一大，参数就乱飞 → 过拟合，所以 LinearRegression 比较“敏感”，容易被数据带偏。

Ridge（L2 正则）—— 让参数更“稳”：
损失函数=误差+a*Wi平方和，不能让权重太大，否则有惩罚。非常常用的一个“稳一点的线性回归”。

Lasso（L1 正则）—— 会自动“砍掉”没用特征：
损失函数=误差+a∑|w|,因为是取绝对值，它有一个效果：会把某些权重强行推成 0，也就是它能自动选特征（Feature Selection），比如 20 个特征里，它可能把 8 个权重砍成 0。保留 12 个有效特征。
"""
MODEL_TYPE = "l2"  # 决定用哪种线性模型，"none"=无正则, "l2"=Ridge, "l1"=Lasso

ALPHA = 1.0  # 正则强度，仅在 l2/l1 时生效

"""
MAE：平均绝对误差（Mean Absolute Error），MAE=平均（|预测-实际|）
RMSE：均方根误差（Root Mean Squared Error），RMSE=√平均(（预测-实际）^2)，RMSE 尤其关注大偏差。
KFold：把训练集切成 K 份，轮流做验证，KFold（k=5）就是训练 5 次，而且每次训练集都不一样，验证集也不一样。
"""
EVAL_MODE = "mae"  # 决定评估方式，可选:"mae"：在固定 train(1000) / val(500) 划分上算 MAE。"rmse"：同理算 RMSE。"cv"：对 1500 条标注数据做 KFold 交叉验证。

CV_FOLDS = 5  # 仅 EVAL_MODE="cv" 生效
SHUFFLE_TRAIN_VAL = True  # 是否打乱后再切1000/500
RANDOM_STATE = 42  # 随机种子（保证可复现）
DO_PREDICT = True  # True=启动预测导出CSV
PRED_OUTPATH = "./data/predictions_linear_regression.csv"  # 预测结果生成在此文件名下


def make_ohe():
    """兼容不同sklearn版本的OneHotEncoder参数名。"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # 新版
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # 旧版


def build_pipeline(model_type: str, alpha: float,
                   numeric_cols, categorical_cols) -> Pipeline:
    """构建预处理+线性模型的pipeline。"""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    if model_type == "none":
        reg = LinearRegression()
    elif model_type == "l2":
        reg = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    elif model_type == "l1":
        reg = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=10000)
    else:
        raise ValueError("MODEL_TYPE must be one of: 'none', 'l2', 'l1'")

    return Pipeline(steps=[("prep", preprocessor), ("reg", reg)])


def evaluate_fixed_split(model: Pipeline,
                         X_train, y_train, X_val, y_val,
                         mode: str) -> float:
    """在固定的train(1000)/val(500)划分上评估，返回数值越小越好。"""
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    if mode == "mae":
        score = mean_absolute_error(y_val, pred)
    elif mode == "rmse":
        score = mean_squared_error(y_val, pred, squared=False)  # RMSE
    else:
        raise ValueError("For fixed split, EVAL_MODE must be 'mae' or 'rmse'.")

    return score


def evaluate_cross_validation(model: Pipeline, X, y,
                              mode: str, folds: int) -> float:
    """在1500条标注数据上做K折交叉验证，返回均值（越小越好）。"""
    if mode == "mae":
        scoring = "neg_mean_absolute_error"
    elif mode == "rmse":
        # 有的版本提供neg_root_mean_squared_error；若不可用，则自定义
        try:
            scoring = "neg_root_mean_squared_error"
            # 先试一下是否可用
            _ = cross_val_score(model, X, y, cv=2, scoring=scoring)
        except Exception:
            scoring = make_scorer(
                lambda yt, yp: -mean_squared_error(yt, yp, squared=False)
            )
    else:
        # 如果选择使用"cv"则生效
        scoring = "neg_root_mean_squared_error"
        try:
            _ = cross_val_score(model, X, y, cv=2, scoring=scoring)
        except Exception:
            scoring = make_scorer(
                lambda yt, yp: -mean_squared_error(yt, yp, squared=False)
            )

    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    # cross_val_score是“越大越好”（因为是负误差），取负号转为“越小越好”
    return float(-scores.mean())


def run_prediction(model: Pipeline, X_pred: pd.DataFrame,
                   pred_df: pd.DataFrame, out_path: str):
    """对预测集生成预测并导出CSV。"""
    y_pred = model.predict(X_pred)
    out = pred_df.copy()
    out["y_pred"] = y_pred
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ 已导出预测结果到: {out_path}")


def main():
    # 1) 读数据
    df = pd.read_csv(DATA_PATH)

    # 2) 目标列：优先y，否则最后一列
    target_col = "y" if "y" in df.columns else df.columns[-1]

    # 保留原始行号，便于回查
    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})

    # 3) 训练/预测划分：有y的为标注集（1500），缺y的为预测集（500）
    is_labeled = ~df[target_col].isna()
    labeled_df = df[is_labeled].copy()
    pred_df = df[~is_labeled].copy()

    # 如果没有缺失y，就按题意把最后500行当预测集
    if pred_df.empty:
        pred_df = df.tail(500).copy()
        labeled_df = df.drop(pred_df.index).copy()

    # 4) 从标注集里取1000/500做 train/val
    drop_cols = {target_col, "row_index", "id", "ID", "Id"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if SHUFFLE_TRAIN_VAL:
        labeled_df = labeled_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    train_df = labeled_df.iloc[:1000].copy()
    val_df = labeled_df.iloc[1000:1500].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(float)

    X_pred = pred_df[feature_cols]

    # 数值 / 类别列
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # 5) 构建模型
    model = build_pipeline(MODEL_TYPE, ALPHA, numeric_cols, categorical_cols)

    # 6) 评估
    print(f"=== 配置 ===")
    print(f"MODEL_TYPE={MODEL_TYPE}, ALPHA={ALPHA}, EVAL_MODE={EVAL_MODE}")
    print(f"Train={len(train_df)}, Val={len(val_df)}, Predict={len(pred_df)}")

    if EVAL_MODE in ("mae", "rmse"):
        score = evaluate_fixed_split(model, X_train, y_train, X_val, y_val, EVAL_MODE)
        print(f"[Fixed 1000/500] {EVAL_MODE.upper()} = {score:.6f} (越小越好)")
    elif EVAL_MODE == "cv":
        # 在1500条标注数据整体上做K折CV
        X_all = labeled_df[feature_cols]
        y_all = labeled_df[target_col].astype(float)
        cv_score = evaluate_cross_validation(model, X_all, y_all, "rmse", CV_FOLDS)
        print(f"[{CV_FOLDS}-Fold CV on 1500 labeled] RMSE = {cv_score:.6f} (越小越好)")
    else:
        raise ValueError("EVAL_MODE must be one of: 'mae', 'rmse', 'cv'.")

    # 7) 预测（可选）
    if DO_PREDICT:
        # 预测前用全部1500标注数据重新拟合（通常这样更充分）
        X_all = labeled_df[feature_cols]
        y_all = labeled_df[target_col].astype(float)
        model.fit(X_all, y_all)
        run_prediction(model, X_pred, pred_df, PRED_OUTPATH)


if __name__ == "__main__":
    main()
