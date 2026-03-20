import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景13: 训练集添加合成数据，测试集 Smile/Finger 真实，Quick 使用合成 (65.2%替换)
# Scenario 13: Synthetic train, real test for Smile & Finger, synthetic test for Quick
# 目标: 评估 Quick synthetic test 的影响
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/synthetic_train_real_smile_finger/features_demography_diagnosis.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/synthetic_train_real_smile_finger/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/synthetic_train_real_smile_finger/facial_dataset.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
    # 1: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal'],
    # 2: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],
    # 3: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal']
}

