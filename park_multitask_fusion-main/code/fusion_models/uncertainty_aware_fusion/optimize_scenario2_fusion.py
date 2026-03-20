#!/usr/bin/env python3
"""
优化Scenario 2的fusion超参数
通过网格搜索找到最佳的模态权重和融合策略
"""
import os
import sys
sys.path.append('/localdisk2/pliu/PARK/code')

import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from itertools import product
import baal.bayesian.dropout as mcdropout
from baal.modelwrapper import ModelWrapper
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score

BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
MODELS_DIR = os.path.join(BASE_DIR, "models")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device} ...")

# Model classes (same as before)
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

def load_smile_data(csv_file):
    """加载smile数据"""
    df = pd.read_csv(csv_file)
    df['ID'] = df['ID'].astype(str)
    
    smile_feature_cols = [col for col in df.columns if 'smile_' in col.lower()]
    df[smile_feature_cols] = df[smile_feature_cols].fillna(0)
    
    features = df[smile_feature_cols].values
    labels = (df['pd'] == 'yes').astype(int).values
    ids = df['ID'].values
    
    return features, labels, ids

def concat_features(row):
    return np.concatenate([row["features_right"], row["features_left"]])

def load_finger_data(csv_file):
    """加载finger tapping数据"""
    df = pd.read_csv(csv_file)
    df['Participant_ID'] = df['Participant_ID'].astype(str)
    
    metadata_cols = ['filename', 'Protocol', 'Participant_ID', 'Task', 'Duration', 'FPS', 
                    'Frame_Height', 'Frame_Width', 'gender', 'age', 'race', 'ethnicity', 
                    'pd', 'dob', 'time_mdsupdrs', 'date', 'hand']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    df_right = df[df['hand'] == 'right'].copy()
    df_left = df[df['hand'] == 'left'].copy()
    
    df_right['features_right'] = df_right[feature_cols].values.tolist()
    df_left['features_left'] = df_left[feature_cols].values.tolist()
    
    df_both = pd.merge(df_right[['Participant_ID', 'features_right', 'pd']], 
                       df_left[['Participant_ID', 'features_left']], 
                       on='Participant_ID', how='inner')
    
    df_both['features'] = df_both.apply(concat_features, axis=1)
    
    features = np.array(df_both['features'].tolist())
    labels = (df_both['pd'] == 'yes').astype(int).values
    ids = df_both['Participant_ID'].values
    
    return features, labels, ids

def load_speech_data(csv_file):
    """加载speech数据"""
    df = pd.read_csv(csv_file)
    df['ID'] = df['ID'].astype(str)
    
    feature_cols = [str(i) for i in range(1024)]
    available_cols = [col for col in feature_cols if col in df.columns]
    
    df[available_cols] = df[available_cols].fillna(0)
    
    features = df[available_cols].values
    labels = (df['pd'] == 'yes').astype(int).values
    ids = df['ID'].values
    
    return features, labels, ids

def load_all_predictions():
    """加载三个模态的预测"""
    print("Loading all modality predictions...")
    
    # Load Smile
    print("\n📊 Loading Smile...")
    smile_features, smile_labels, smile_ids = load_smile_data(
        "/localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/chinese_smile_real.csv"
    )
    
    model_dir = os.path.join(MODELS_DIR, "facial_expression_smile_best_auroc_baal")
    with open(os.path.join(model_dir, "predictive_model/model_config.json"), 'r') as f:
        config = json.load(f)
    
    scaler = pickle.load(open(os.path.join(model_dir, "scaler/scaler.pth"), 'rb'))
    use_scaling = config.get('use_feature_scaling', 'no') == 'yes'
    
    if config['model'] == 'ShallowANN':
        model = ShallowANN(smile_features.shape[1], config['dropout_prob'])
    else:
        model = ANN(smile_features.shape[1], config['dropout_prob'])
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "predictive_model/model.pth"), map_location=device))
    model = model.to(device)
    
    smile_preds, smile_uncerts = predict_with_model(model, scaler, smile_features, use_scaling, num_trials=300)
    
    # Load Finger
    print("\n📊 Loading Finger...")
    finger_features, finger_labels, finger_ids = load_finger_data(
        "/localdisk2/pliu/park_multitask_fusion-main/data/chinese_finger_real.csv"
    )
    
    model_dir = os.path.join(MODELS_DIR, "finger_model_both_hand_fusion_baal")
    with open(os.path.join(model_dir, "predictive_model/model_config.json"), 'r') as f:
        config = json.load(f)
    
    scaler = pickle.load(open(os.path.join(model_dir, "scaler/scaler.pth"), 'rb'))
    use_scaling = config.get('use_feature_scaling', 'no') == 'yes'
    
    if config['model'] == 'ShallowANN':
        model = ShallowANN(finger_features.shape[1], config['dropout_prob'])
    else:
        model = ANN(finger_features.shape[1], config['dropout_prob'])
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "predictive_model/model.pth"), map_location=device))
    model = model.to(device)
    
    finger_preds, finger_uncerts = predict_with_model(model, scaler, finger_features, use_scaling, num_trials=300)
    
    # Load Speech
    print("\n📊 Loading Speech...")
    speech_features, speech_labels, speech_ids = load_speech_data(
        "/localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/scenario2_smile_finger/chinese_quick_synthetic.csv"
    )
    
    model_dir = os.path.join(MODELS_DIR, "fox_model_best_auroc_baal")
    with open(os.path.join(model_dir, "predictive_model/model_config.json"), 'r') as f:
        config = json.load(f)
    
    scaler = pickle.load(open(os.path.join(model_dir, "scaler/scaler.pth"), 'rb'))
    use_scaling = config.get('use_feature_scaling', 'no') == 'yes'
    
    if config['model'] == 'ShallowANN':
        model = ShallowANN(speech_features.shape[1], config['dropout_prob'])
    else:
        model = ANN(speech_features.shape[1], config['dropout_prob'])
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "predictive_model/model.pth"), map_location=device))
    model = model.to(device)
    
    speech_preds, speech_uncerts = predict_with_model(model, scaler, speech_features, use_scaling, num_trials=300)
    
    # Find common IDs
    common_ids = set(smile_ids).intersection(set(finger_ids)).intersection(set(speech_ids))
    print(f"\n🔗 Common participants: {len(common_ids)}")
    
    # Organize by ID
    data = {}
    for pid in common_ids:
        smile_idx = list(smile_ids).index(pid)
        finger_idx = list(finger_ids).index(pid)
        speech_idx = list(speech_ids).index(pid)
        
        data[pid] = {
            'smile_pred': smile_preds[smile_idx],
            'smile_uncert': smile_uncerts[smile_idx],
            'finger_pred': finger_preds[finger_idx],
            'finger_uncert': finger_uncerts[finger_idx],
            'speech_pred': speech_preds[speech_idx],
            'speech_uncert': speech_uncerts[speech_idx],
            'label': smile_labels[smile_idx]
        }
    
    return data

def test_fusion_strategy(data, w_smile, w_finger, w_speech, threshold=0.5, use_uncertainty=False):
    """测试一个fusion策略"""
    fused_preds = []
    labels = []
    
    for pid, d in data.items():
        if use_uncertainty:
            # 基于不确定性的加权
            u_smile = 1.0 / (1.0 + d['smile_uncert'])
            u_finger = 1.0 / (1.0 + d['finger_uncert'])
            u_speech = 1.0 / (1.0 + d['speech_uncert'])
            total_u = u_smile + u_finger + u_speech
            
            fused_pred = (u_smile * d['smile_pred'] + 
                         u_finger * d['finger_pred'] + 
                         u_speech * d['speech_pred']) / total_u
        else:
            # 固定权重
            total_w = w_smile + w_finger + w_speech
            fused_pred = (w_smile * d['smile_pred'] + 
                         w_finger * d['finger_pred'] + 
                         w_speech * d['speech_pred']) / total_w
        
        fused_preds.append(fused_pred)
        labels.append(d['label'])
    
    metrics = compute_metrics(labels, fused_preds, threshold)
    return metrics

def grid_search():
    """网格搜索最佳超参数"""
    print("\n" + "="*80)
    print("Grid Search for Optimal Fusion Hyperparameters (Scenario 2)")
    print("="*80)
    
    # Load predictions
    data = load_all_predictions()
    
    # Define search space
    weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_options = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    best_bal_acc = 0
    best_f1 = 0
    best_params_bal_acc = None
    best_params_f1 = None
    
    results = []
    
    print("\n🔍 Testing weight combinations...")
    total_combinations = len(weight_options) ** 3 * len(threshold_options)
    print(f"Total combinations to test: {total_combinations}")
    
    count = 0
    for w_s in weight_options:
        for w_f in weight_options:
            for w_sp in weight_options:
                for thr in threshold_options:
                    count += 1
                    if count % 500 == 0:
                        print(f"Progress: {count}/{total_combinations}")
                    
                    metrics = test_fusion_strategy(data, w_s, w_f, w_sp, threshold=thr)
                    
                    result = {
                        'w_smile': w_s,
                        'w_finger': w_f,
                        'w_speech': w_sp,
                        'threshold': thr,
                        'balanced_accuracy': metrics['balanced_accuracy'],
                        'f1_score': metrics['f1_score'],
                        'auroc': metrics['auroc'],
                        'accuracy': metrics['accuracy'],
                        'sensitivity': metrics['sensitivity'],
                        'specificity': metrics['specificity']
                    }
                    results.append(result)
                    
                    if metrics['balanced_accuracy'] > best_bal_acc:
                        best_bal_acc = metrics['balanced_accuracy']
                        best_params_bal_acc = result.copy()
                        best_params_bal_acc['full_metrics'] = metrics
                    
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_params_f1 = result.copy()
                        best_params_f1['full_metrics'] = metrics
    
    # Test uncertainty-based fusion
    print("\n🔍 Testing uncertainty-based fusion...")
    for thr in threshold_options:
        metrics = test_fusion_strategy(data, 0, 0, 0, threshold=thr, use_uncertainty=True)
        
        result = {
            'strategy': 'uncertainty_based',
            'threshold': thr,
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1_score': metrics['f1_score'],
            'auroc': metrics['auroc'],
            'accuracy': metrics['accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity']
        }
        
        if metrics['balanced_accuracy'] > best_bal_acc:
            best_bal_acc = metrics['balanced_accuracy']
            best_params_bal_acc = result.copy()
            best_params_bal_acc['full_metrics'] = metrics
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_params_f1 = result.copy()
            best_params_f1['full_metrics'] = metrics
    
    # Print results
    print("\n" + "="*80)
    print("BEST RESULTS")
    print("="*80)
    
    print("\n🏆 Best Balanced Accuracy Configuration:")
    print(f"  Strategy: {best_params_bal_acc.get('strategy', 'weighted')}")
    if 'w_smile' in best_params_bal_acc:
        print(f"  Weights: Smile={best_params_bal_acc['w_smile']:.1f}, "
              f"Finger={best_params_bal_acc['w_finger']:.1f}, "
              f"Speech={best_params_bal_acc['w_speech']:.1f}")
    print(f"  Threshold: {best_params_bal_acc['threshold']:.2f}")
    print(f"  Balanced Accuracy: {best_params_bal_acc['balanced_accuracy']:.4f}")
    print(f"  F1 Score: {best_params_bal_acc['f1_score']:.4f}")
    print(f"  AUROC: {best_params_bal_acc['auroc']:.4f}")
    print(f"  Accuracy: {best_params_bal_acc['accuracy']:.4f}")
    print(f"  Sensitivity: {best_params_bal_acc['sensitivity']:.4f}")
    print(f"  Specificity: {best_params_bal_acc['specificity']:.4f}")
    
    print("\n🏆 Best F1 Score Configuration:")
    print(f"  Strategy: {best_params_f1.get('strategy', 'weighted')}")
    if 'w_smile' in best_params_f1:
        print(f"  Weights: Smile={best_params_f1['w_smile']:.1f}, "
              f"Finger={best_params_f1['w_finger']:.1f}, "
              f"Speech={best_params_f1['w_speech']:.1f}")
    print(f"  Threshold: {best_params_f1['threshold']:.2f}")
    print(f"  Balanced Accuracy: {best_params_f1['balanced_accuracy']:.4f}")
    print(f"  F1 Score: {best_params_f1['f1_score']:.4f}")
    print(f"  AUROC: {best_params_f1['auroc']:.4f}")
    print(f"  Accuracy: {best_params_f1['accuracy']:.4f}")
    print(f"  Sensitivity: {best_params_f1['sensitivity']:.4f}")
    print(f"  Specificity: {best_params_f1['specificity']:.4f}")
    
    # Compare with original
    print("\n" + "="*80)
    print("COMPARISON WITH ORIGINAL SCENARIO 2")
    print("="*80)
    
    original_metrics = test_fusion_strategy(data, 1.0, 1.0, 1.0, threshold=0.5)
    print("\n📊 Original (Equal weights, threshold=0.5):")
    print(f"  Balanced Accuracy: {original_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Score: {original_metrics['f1_score']:.4f}")
    print(f"  AUROC: {original_metrics['auroc']:.4f}")
    
    print(f"\n📈 Improvement (Best Bal. Acc):")
    print(f"  Balanced Accuracy: {original_metrics['balanced_accuracy']:.4f} → "
          f"{best_params_bal_acc['balanced_accuracy']:.4f} "
          f"(+{(best_params_bal_acc['balanced_accuracy'] - original_metrics['balanced_accuracy'])*100:.2f}%)")
    print(f"  F1 Score: {original_metrics['f1_score']:.4f} → "
          f"{best_params_bal_acc['f1_score']:.4f} "
          f"(+{(best_params_bal_acc['f1_score'] - original_metrics['f1_score'])*100:.2f}%)")
    
    # Save results
    output = {
        'best_balanced_accuracy': best_params_bal_acc,
        'best_f1_score': best_params_f1,
        'original': original_metrics,
        'all_results': results[:100]  # Save top 100
    }
    
    output_file = os.path.join(BASE_DIR, "data/scenario2_optimized_params.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    return best_params_bal_acc, best_params_f1

if __name__ == "__main__":
    grid_search()

