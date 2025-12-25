"""
Configurações globais do projeto de previsão Mega-Sena.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "Mega-Sena.csv"
MODELS_PATH = PROJECT_ROOT / "models_saved"
RESULTS_PATH = PROJECT_ROOT / "results"

# Data columns
BALL_COLUMNS = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
DATE_COLUMN = 'Data do Sorteio'
CONTEST_COLUMN = 'Concurso'

# Mega-Sena parameters
TOTAL_NUMBERS = 60
NUMBERS_PER_DRAW = 6
TOTAL_COMBINATIONS = 50_063_860

# Model parameters
ARIMA_ORDER = (5, 1, 0)
LSTM_LOOKBACK = 10
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_UNITS = [100, 50]
DROPOUT_RATE = 0.2

# Evaluation
TOP_K_VALUES = [10, 15, 20, 25, 30]
TRAIN_TEST_SPLIT = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42
