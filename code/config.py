seed = 42
audioSampleRate = 16000

# Data Split 
n_splits = 10
slidingWindows = 2.5
step = 1.5

# Feature Extraction
FFTwindow = 256
FFTOverlap = 128
MFCCFiliterNum = 26 

model_type = 'lgbm' # model type: 'svm', 'rf', 'lgbm'

dataDir = '/data/Leo/mm/data/NICU50/data/'
# dataDir = '/data/Leo/mm/data/NEWBORN200/data/'

# import os
# # Project root directory
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Data directories: NEWBORN200 NICU50
# dataDir = os.path.join(PROJECT_ROOT, 'dataset', 'NICU50', 'data')