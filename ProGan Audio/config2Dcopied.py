import torch
from math import log2

START_TRAIN_AT_SEQ_LEN = 256  # arrondi à la puissance de 2 la plus proche

DATASET = r"A:\Program Files\Documents infaudio\AI_audio\WaveGan\drums\train"

CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = False
LOAD_MODEL = False
LEARNING_RATE = 2e-4  # plus stable pour GAN

# Batch sizes adaptés à la taille des séquences (progressive training)
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 4, 2, 1]

CHANNELS_AUDIO = 1  # mono audio (1 canal)
Z_DIM = 256
IN_CHANNELS = 16

CRITIC_ITERATIONS = 2
LAMBDA_GP = 10

PROGRESSIVE_EPOCHS = [7] * len(BATCH_SIZES)

FIXED_NOISE = torch.randn(8, Z_DIM, 1).to(DEVICE)

NUM_WORKERS = 0
