import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景11: 训练集添加合成数据，测试集 Finger/Quick 真实，Smile 使用合成 (62.3%替换)
# Scenario 11: Synthetic train, real test for Finger & Quick, synthetic test for Smile
# 目标: 评估 Smile synthetic test 的影响
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/synthetic_train_real_finger_quick/features_demography_diagnosis.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/synthetic_train_real_finger_quick/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/synthetic_train_real_finger_quick/facial_dataset.csv")

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

