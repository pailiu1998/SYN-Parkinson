#!/usr/bin/env python3
"""
优化所有fusion场景的超参数
包括:
1. Scenario 1: Real Smile + Synthetic Finger + Synthetic Speech
2. Scenario 2: Real Smile + Real Finger + Synthetic Speech
3. Scenario 3: Real Smile + Synthetic Speech (bimodal)
4. Real Bimodal: Real Smile + Real Finger (bimodal)
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

def load_modality_predictions(modality_name, csv_file, model_name):
    """加载单个模态的预测"""
    print(f"  Loading {modality_name}...")
    
    if 'smile' in modality_name.lower():
        features, labels, ids = load_smile_data(csv_file)
    elif 'finger' in modality_name.lower():
        features, labels, ids = load_finger_data(csv_file)
    elif 'speech' in modality_name.lower() or 'quick' in modality_name.lower():
        features, labels, ids = load_speech_data(csv_file)
    else:
        raise ValueError(f"Unknown modality: {modality_name}")
    
    model_dir = os.path.join(MODELS_DIR, model_name)
    with open(os.path.join(model_dir, "predictive_model/model_config.json"), 'r') as f:
        config = json.load(f)
    
    scaler = pickle.load(open(os.path.join(model_dir, "scaler/scaler.pth"), 'rb'))
    use_scaling = config.get('use_feature_scaling', 'no') == 'yes'
    
    if config['model'] == 'ShallowANN':
        model = ShallowANN(features.shape[1], config['dropout_prob'])
    else:
        model = ANN(features.shape[1], config['dropout_prob'])
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "predictive_model/model.pth"), map_location=device))
    model = model.to(device)
    
    preds, uncerts = predict_with_model(model, scaler, features, use_scaling, num_trials=300)
    
    return preds, uncerts, labels, ids

def optimize_scenario(scenario_name, modalities_config):
    """优化一个scenario"""
    print(f"\n{'='*80}")
    print(f"Optimizing {scenario_name}")
    print(f"{'='*80}")
    
    # Load all modality predictions
    all_preds = {}
    all_uncerts = {}
    all_labels = None
    all_ids = None
    
    for mod_name, (csv_file, model_name) in modalities_config.items():
        preds, uncerts, labels, ids = load_modality_predictions(mod_name, csv_file, model_name)
        all_preds[mod_name] = preds
        all_uncerts[mod_name] = uncerts
        
        if all_labels is None:
            all_labels = labels
            all_ids = set(ids)
        else:
            all_ids = all_ids.intersection(set(ids))
    
    # Organize by common IDs
    common_ids = list(all_ids)
    print(f"  Common participants: {len(common_ids)}")
    
    data = {}
    for pid in common_ids:
        data[pid] = {'label': None}
        for mod_name, csv_file_model in modalities_config.items():
            csv_file, _ = csv_file_model
            
            if 'smile' in mod_name.lower():
                _, labels, ids = load_smile_data(csv_file)
            elif 'finger' in mod_name.lower():
                _, labels, ids = load_finger_data(csv_file)
            else:
                _, labels, ids = load_speech_data(csv_file)
            
            idx = list(ids).index(pid)
            data[pid][f'{mod_name}_pred'] = all_preds[mod_name][idx]
            data[pid][f'{mod_name}_uncert'] = all_uncerts[mod_name][idx]
            data[pid]['label'] = labels[idx]
    
    # Grid search
    modality_names = list(modalities_config.keys())
    num_modalities = len(modality_names)
    
    if num_modalities == 2:
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_options = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    else:  # 3 modalities
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_options = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    best_f1 = 0
    best_bal_acc = 0
    best_params_f1 = None
    best_params_bal_acc = None
    
    print(f"  Grid searching...")
    
    if num_modalities == 2:
        for w1 in weight_options:
            for w2 in weight_options:
                for thr in threshold_options:
                    # Fusion
                    fused_preds = []
                    labels = []
                    
                    for pid, d in data.items():
                        total_w = w1 + w2
                        fused_pred = (w1 * d[f'{modality_names[0]}_pred'] + 
                                     w2 * d[f'{modality_names[1]}_pred']) / total_w
                        fused_preds.append(fused_pred)
                        labels.append(d['label'])
                    
                    metrics = compute_metrics(labels, fused_preds, thr)
                    
                    result = {
                        'weights': {modality_names[0]: w1, modality_names[1]: w2},
                        'threshold': thr,
                        'metrics': metrics
                    }
                    
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_params_f1 = result
                    
                    if metrics['balanced_accuracy'] > best_bal_acc:
                        best_bal_acc = metrics['balanced_accuracy']
                        best_params_bal_acc = result
    
    else:  # 3 modalities
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    for thr in threshold_options:
                        # Fusion
                        fused_preds = []
                        labels = []
                        
                        for pid, d in data.items():
                            total_w = w1 + w2 + w3
                            fused_pred = (w1 * d[f'{modality_names[0]}_pred'] + 
                                         w2 * d[f'{modality_names[1]}_pred'] +
                                         w3 * d[f'{modality_names[2]}_pred']) / total_w
                            fused_preds.append(fused_pred)
                            labels.append(d['label'])
                        
                        metrics = compute_metrics(labels, fused_preds, thr)
                        
                        result = {
                            'weights': {modality_names[0]: w1, modality_names[1]: w2, modality_names[2]: w3},
                            'threshold': thr,
                            'metrics': metrics
                        }
                        
                        if metrics['f1_score'] > best_f1:
                            best_f1 = metrics['f1_score']
                            best_params_f1 = result
                        
                        if metrics['balanced_accuracy'] > best_bal_acc:
                            best_bal_acc = metrics['balanced_accuracy']
                            best_params_bal_acc = result
    
    # Compute original (equal weights)
    fused_preds = []
    labels = []
    for pid, d in data.items():
        pred_sum = sum([d[f'{mod}_pred'] for mod in modality_names])
        fused_pred = pred_sum / num_modalities
        fused_preds.append(fused_pred)
        labels.append(d['label'])
    
    original_metrics = compute_metrics(labels, fused_preds, 0.5)
    
    return {
        'scenario_name': scenario_name,
        'best_f1': best_params_f1,
        'best_balanced_accuracy': best_params_bal_acc,
        'original': original_metrics,
        'num_participants': len(common_ids)
    }

def main():
    print("\n" + "="*80)
    print("Comprehensive Fusion Hyperparameter Optimization")
    print("="*80)
    
    # Define all scenarios
    scenarios = {
        'Scenario 1 (S+Fsyn+Spsyn)': {
            'Smile': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/chinese_smile_real.csv"),
                     "facial_expression_smile_best_auroc_baal"),
            'Finger': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario1_smile_only/chinese_finger_synthetic.csv"),
                      "finger_model_both_hand_fusion_baal"),
            'Speech': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario1_smile_only/chinese_quick_synthetic.csv"),
                      "fox_model_best_auroc_baal")
        },
        'Scenario 2 (S+F+Spsyn)': {
            'Smile': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/chinese_smile_real.csv"),
                     "facial_expression_smile_best_auroc_baal"),
            'Finger': (os.path.join(BASE_DIR, "data/chinese_finger_real.csv"),
                      "finger_model_both_hand_fusion_baal"),
            'Speech': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario2_smile_finger/chinese_quick_synthetic.csv"),
                      "fox_model_best_auroc_baal")
        },
        'Scenario 3 (S+Spsyn)': {
            'Smile': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/chinese_smile_real.csv"),
                     "facial_expression_smile_best_auroc_baal"),
            'Speech': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario2_smile_finger/chinese_quick_synthetic.csv"),
                      "fox_model_best_auroc_baal")
        },
        'Real Bimodal (S+F)': {
            'Smile': (os.path.join(BASE_DIR, "data/chinese_synthetic_data/chinese_smile_real.csv"),
                     "facial_expression_smile_best_auroc_baal"),
            'Finger': (os.path.join(BASE_DIR, "data/chinese_finger_real.csv"),
                      "finger_model_both_hand_fusion_baal")
        }
    }
    
    # Run optimization for each scenario
    all_results = {}
    for scenario_name, config in scenarios.items():
        result = optimize_scenario(scenario_name, config)
        all_results[scenario_name] = result
    
    # Save results
    output_file = os.path.join(BASE_DIR, "data/all_scenarios_optimized.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    for scenario_name, result in all_results.items():
        print(f"\n{scenario_name}")
        print(f"  Participants: {result['num_participants']}")
        
        print(f"\n  Original (Equal Weights, threshold=0.5):")
        print(f"    AUROC: {result['original']['auroc']:.4f}")
        print(f"    Accuracy: {result['original']['accuracy']:.4f}")
        print(f"    F1 Score: {result['original']['f1_score']:.4f}")
        print(f"    Bal. Acc: {result['original']['balanced_accuracy']:.4f}")
        
        best = result['best_balanced_accuracy']
        print(f"\n  Optimized (Best Balanced Accuracy):")
        print(f"    Weights: {best['weights']}")
        print(f"    Threshold: {best['threshold']:.2f}")
        print(f"    AUROC: {best['metrics']['auroc']:.4f}")
        print(f"    Accuracy: {best['metrics']['accuracy']:.4f}")
        print(f"    F1 Score: {best['metrics']['f1_score']:.4f}")
        print(f"    Bal. Acc: {best['metrics']['balanced_accuracy']:.4f}")
        print(f"    Sensitivity: {best['metrics']['sensitivity']:.4f}")
        print(f"    Specificity: {best['metrics']['specificity']:.4f}")
        
        print(f"\n  Improvement:")
        print(f"    AUROC: +{(best['metrics']['auroc'] - result['original']['auroc'])*100:.2f}%")
        print(f"    Bal. Acc: +{(best['metrics']['balanced_accuracy'] - result['original']['balanced_accuracy'])*100:.2f}%")
        print(f"    F1 Score: +{(best['metrics']['f1_score'] - result['original']['f1_score'])*100:.2f}%")

if __name__ == "__main__":
    main()

