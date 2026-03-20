#!/usr/bin/env python3
"""
使用已有的预训练unimodal模型在中文数据上测试
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import baal.bayesian.dropout as mcdropout
from baal.modelwrapper import ModelWrapper
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, recall_score, precision_score

BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
MODELS_DIR = os.path.join(BASE_DIR, "models")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device} ...")

# Model classes
class TensorDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.Tensor(np.asarray(features))
        self.labels = torch.Tensor(labels) if labels is not None else None

    def __getitem__(self, index):
        if self.labels is not None:
            return self.features[index], self.labels[index]
        else:
            return self.features[index], torch.tensor(0.0)

    def __len__(self):
        return len(self.features)

class ShallowANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ShallowANN, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
        self.drop = mcdropout.Dropout(p=drop_prob)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        y = self.drop(y)
        y = self.sig(y)
        return y

class ANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=int(n_features/2), bias=True)
        self.drop1 = mcdropout.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1, bias=True)
        self.drop2 = mcdropout.Dropout(p=drop_prob)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop1(x1)
        y = self.fc2(x1)
        y = self.drop2(y)
        y = self.sig(y)
        return y

def compute_metrics(y_true, y_pred_scores, threshold=0.5):
    """计算评估指标"""
    labels = np.asarray(y_true).reshape(-1)
    pred_scores = np.asarray(y_pred_scores).reshape(-1)
    preds = (pred_scores >= threshold)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['auroc'] = roc_auc_score(labels, pred_scores)
    metrics['f1_score'] = f1_score(labels, preds, zero_division=0)
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2.0
    
    return metrics

def predict_with_model(model, scaler, features, use_scaling, num_trials=300, batch_size=32):
    """使用模型进行预测"""
    if use_scaling and scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    dataset = TensorDataset(features_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    criterion = torch.nn.BCELoss()
    wrapped_model = ModelWrapper(model, criterion)
    
    all_preds = []
    all_uncertainties = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            y_multi_preds = wrapped_model.predict_on_batch(x, iterations=num_trials)
            y_preds = y_multi_preds.mean(dim=-1)
            y_errors = y_multi_preds.std(dim=-1)
            
            all_preds.extend(y_preds.cpu().numpy())
            all_uncertainties.extend(y_errors.cpu().numpy())
    
    return np.array(all_preds), np.array(all_uncertainties)

def load_smile_data_chinese(csv_file):
    """加载中文smile数据"""
    df = pd.read_csv(csv_file)
    df['ID'] = df['ID'].astype(str)
    
    smile_feature_cols = [col for col in df.columns if 'smile_' in col.lower()]
    df[smile_feature_cols] = df[smile_feature_cols].fillna(0)
    
    features = df[smile_feature_cols].values
    labels = (df['pd'] == 'yes').astype(int).values
    ids = df['ID'].values
    
    return features, labels, ids

def load_finger_data_chinese(csv_file):
    """加载中文finger tapping数据"""
    df = pd.read_csv(csv_file)
    df['Participant_ID'] = df['Participant_ID'].astype(str)
    
    metadata_cols = ['filename', 'Protocol', 'Participant_ID', 'Task', 'Duration', 'FPS', 
                    'Frame_Height', 'Frame_Width', 'gender', 'age', 'race', 'ethnicity', 
                    'pd', 'dob', 'time_mdsupdrs', 'date', 'hand']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # 分离左右手
    df_right = df[df['hand'] == 'right'].copy()
    df_left = df[df['hand'] == 'left'].copy()
    
    df_right['features_right'] = df_right[feature_cols].values.tolist()
    df_left['features_left'] = df_left[feature_cols].values.tolist()
    
    # 合并左右手
    df_both = pd.merge(
        df_right[['Participant_ID', 'features_right', 'pd']], 
        df_left[['Participant_ID', 'features_left']], 
        on='Participant_ID', 
        how='inner'
    )
    
    # 拼接特征
    features_list = []
    for _, row in df_both.iterrows():
        feat = np.concatenate([np.asarray(row['features_right']), np.asarray(row['features_left'])])
        features_list.append(feat)
    
    features = np.asarray(features_list)
    labels = (df_both['pd'] == 'yes').astype(int).values
    ids = df_both['Participant_ID'].values
    
    return features, labels, ids

def test_smile_model():
    """测试Smile模型"""
    print("\n" + "="*80)
    print("Testing Smile Model on Chinese Data")
    print("="*80)
    
    # 加载中文数据
    csv_file = os.path.join(BASE_DIR, "data/chinese_smile_real.csv")
    features, labels, ids = load_smile_data_chinese(csv_file)
    print(f"Loaded {len(features)} samples, {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    
    # 加载预训练模型
    model_dir = os.path.join(MODELS_DIR, "facial_expression_smile_best_auroc_baal")
    with open(os.path.join(model_dir, "predictive_model/model_config.json"), 'r') as f:
        config = json.load(f)
    
    print(f"Model config: {config}")
    
    # 加载scaler
    scaler = pickle.load(open(os.path.join(model_dir, "scaler/scaler.pth"), 'rb'))
    use_scaling = config.get('use_feature_scaling', 'no') == 'yes'
    
    # 构建模型
    if config['model'] == 'ShallowANN':
        model = ShallowANN(features.shape[1], config['dropout_prob'])
    else:
        model = ANN(features.shape[1], config['dropout_prob'])
    
    # 加载权重
    model.load_state_dict(torch.load(os.path.join(model_dir, "predictive_model/model.pth"), map_location=device))
    model = model.to(device)
    
    # 预测
    print("Making predictions...")
    preds, uncerts = predict_with_model(model, scaler, features, use_scaling, num_trials=300)
    
    # 计算指标
    metrics = compute_metrics(labels, preds)
    
    print("\n📊 Results:")
    print(f"  AUROC:            {metrics['auroc']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  F1 Score:         {metrics['f1_score']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:      {metrics['sensitivity']:.4f}")
    print(f"  Specificity:      {metrics['specificity']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}")
    print(f"    FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")
    
    return metrics

def test_finger_model():
    """测试Finger模型"""
    print("\n" + "="*80)
    print("Testing Finger Model on Chinese Data")
    print("="*80)
    
    # 加载中文数据
    csv_file = os.path.join(BASE_DIR, "data/chinese_finger_real.csv")
    features, labels, ids = load_finger_data_chinese(csv_file)
    print(f"Loaded {len(features)} samples, {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    print(f"Feature shape: {features.shape}")
    
    # 加载预训练模型
    model_dir = os.path.join(MODELS_DIR, "finger_model_both_hand_fusion_baal")
    with open(os.path.join(model_dir, "predictive_model/model_config.json"), 'r') as f:
        config = json.load(f)
    
    print(f"Model config: {config}")
    
    # 加载scaler
    scaler = pickle.load(open(os.path.join(model_dir, "scaler/scaler.pth"), 'rb'))
    use_scaling = config.get('use_feature_scaling', 'no') == 'yes'
    
    # 构建模型
    if config['model'] == 'ShallowANN':
        model = ShallowANN(features.shape[1], config['dropout_prob'])
    else:
        model = ANN(features.shape[1], config['dropout_prob'])
    
    # 加载权重
    model.load_state_dict(torch.load(os.path.join(model_dir, "predictive_model/model.pth"), map_location=device))
    model = model.to(device)
    
    # 预测
    print("Making predictions...")
    preds, uncerts = predict_with_model(model, scaler, features, use_scaling, num_trials=300)
    
    # 计算指标
    metrics = compute_metrics(labels, preds)
    
    print("\n📊 Results:")
    print(f"  AUROC:            {metrics['auroc']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  F1 Score:         {metrics['f1_score']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:      {metrics['sensitivity']:.4f}")
    print(f"  Specificity:      {metrics['specificity']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}")
    print(f"    FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")
    
    return metrics

if __name__ == "__main__":
    smile_metrics = test_smile_model()
    finger_metrics = test_finger_model()
    
    # 保存结果
    results = {
        'smile': smile_metrics,
        'finger': finger_metrics
    }
    
    output_file = os.path.join(BASE_DIR, "data/chinese_unimodal_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # 打印LaTeX表格
    print("\n" + "="*80)
    print("LaTeX Table")
    print("="*80)
    print(r"""
\begin{table}[h]
\centering
\caption{中文数据单模态测试结果（使用预训练英文模型）}
\label{tab:chinese_unimodal_pretrained}
\begin{tabular}{lcccccc}
\hline
\textbf{模型} & \textbf{AUROC} & \textbf{Accuracy} & \textbf{F1} & \textbf{Bal. Acc} & \textbf{Sens.} & \textbf{Spec.} \\
\hline""")
    
    print(f"Smile (Pretrained) & {smile_metrics['auroc']:.4f} & {smile_metrics['accuracy']:.4f} & "
          f"{smile_metrics['f1_score']:.4f} & {smile_metrics['balanced_accuracy']:.4f} & "
          f"{smile_metrics['sensitivity']:.4f} & {smile_metrics['specificity']:.4f} \\\\")
    
    print(f"Finger (Pretrained) & {finger_metrics['auroc']:.4f} & {finger_metrics['accuracy']:.4f} & "
          f"{finger_metrics['f1_score']:.4f} & {finger_metrics['balanced_accuracy']:.4f} & "
          f"{finger_metrics['sensitivity']:.4f} & {finger_metrics['specificity']:.4f} \\\\")
    
    print(r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item 使用在英文数据上预训练的模型，直接在中文数据上测试（zero-shot transfer）
\item Bal. Acc=Balanced Accuracy, Sens.=Sensitivity, Spec.=Specificity
\end{tablenotes}
\end{table}
""")

