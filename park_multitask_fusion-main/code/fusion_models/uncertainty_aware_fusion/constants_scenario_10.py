import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景10: 在训练数据中添加合成数据（新行方式）
# Scenario 10: Add synthetic data to training data with new rows
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/features_demography_diagnosis.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/facial_dataset.csv")

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


