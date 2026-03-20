import os

# Chinese Real Bimodal Fusion: Smile + Finger
# 中文真实数据双模态融合实验配置

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# 数据路径 - 使用合并后的中英文数据
FINGER_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/wavlm_fox_features.csv")

# 模型路径
MODEL_BASE_PATH = os.path.join(BASE_DIR, "models")

# 面部表情配置
FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

# 三模态配置：Smile + Finger + Syn Speech
MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal', 'fox_model_best_auroc_baal'],
}

# 数据集划分文件路径
DEV_SET_PARTICIPANTS_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/data/dev_set_participants.txt")
TEST_SET_PARTICIPANTS_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/data/test_set_participants.txt")

# 输出路径
RESULTS_DIR = os.path.join(BASE_DIR, "results/chinese_real_bimodal_smile_finger_syn_speech")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 实验配置
EXPERIMENT_NAME = "chinese_real_bimodal_smile_finger_syn_speech"
RANDOM_SEED = 42
NUM_EPOCHS = 244
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.25

print("="*80)
print("Chinese Bimodal Fusion Configuration Loaded")
print("="*80)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Modalities: Smile + Finger (Bimodal)")
print(f"Finger Data: {os.path.basename(FINGER_FEATURES_FILE)}")
print(f"Smile Data: {os.path.basename(FACIAL_FEATURES_FILE)}")
print(f"Models: {MODEL_SUBSETS[0]}")
print(f"Results Dir: {RESULTS_DIR}")
print("="*80)
