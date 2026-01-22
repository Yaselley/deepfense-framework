import os
import yaml
import copy

# --- Constants ---
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/experiments/batch7_PartialSpoof"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base Paths
TRAIN_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/PartialSpoof/train.parquet"
VAL_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/PartialSpoof/dev.parquet"
EVAL_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/PartialSpoof/eval.parquet"

# Frontend Checkpoints (Placeholders based on common paths, user can update)
CKPT_WAV2VEC2 = "/netscratch/yelkheir/ssl_models/xlsr2_300m.pt"
CKPT_HUBERT = "/netscratch/yelkheir/ssl_models/hubert_large_ll60k.pt"
CKPT_WAVLM = "/netscratch/yelkheir/ssl_models/WavLM-Large.pt"

# --- Parameters ---
FRONTENDS = [
    {"type": "wav2vec2", "args": {"ckpt_path": CKPT_WAV2VEC2, "freeze": False, "source": "fairseq"}},
    {"type": "hubert", "args": {"ckpt_path": CKPT_HUBERT, "freeze": False, "source": "fairseq"}},
    {"type": "wavlm", "args": {"ckpt_path": CKPT_WAVLM, "freeze": False, "source": "unil"}},
    {"type": "eat", "args": {"source": "huggingface", "ckpt_path": "worstchan/EAT-large_epoch20_pretrain", "freeze": False}}
]

BACKENDS = [
    {"type": "AASIST", "args": {"input_dim": 1024, "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]], "gat_dims": [64, 32]}},
    {"type": "MLP", "args": {"input_dim": 1024, "projection": [128], "pooling_type": "mean"}},
    {"type": "Nes2Net", "args": {"input_dim": 1024, "filts": [32, 64, 128, 256], "strides": [1, 2, 2, 2]}},
    {"type": "TCM", "args": {"input_dim": 1024, "emb_size": 128, "heads": 4, "num_encoders": 1}}
]

SEEDS = [2, 42, 240]

AUGMENTATIONS = [
    {"name": "NoAug", "augment_transform": []}]

# --- Template ---
BASE_CONFIG = {
    "exp_name": "",
    "output_dir": "./outputs/batch7_PartialSpoof/",
    "seed": 42,
    "data": {
        "sampling_rate": 16000,
        "label_map": {"bonafide": 1, "spoof": 0},
        "train": {
            "dataset_type": "StandardDataset",
            "parquet_files": [TRAIN_PARQUET],
            "dataset_names": ["PartialSpoof_Train"],
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 4,
        },
        "val": {
            "dataset_type": "StandardDataset",
            "parquet_files": [VAL_PARQUET],
            "dataset_names": ["PartialSpoof_Val"],
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 4,
        },
        "test": {
            "dataset_type": "StandardDataset",
            "parquet_files": [EVAL_PARQUET],
            "dataset_names": ["PartialSpoof_Eval"],
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 4,
        }
    },
    "model": {
        "type": "StandardDetector",
        "frontend": {},
        "backend": {},
        "loss": [
            {
                "type": "CrossEntropy",
                "embedding_dim": 128, # Default, adjusted below if backend changes
                "n_classes": 2,
                "weight": 1.0
            }
        ]
    },
    "training": {
        "trainer": "StandardTrainer",
        "epochs": 100,
        "early_stopping_patience": 7,
        "optimizer": {"type": "adam", "lr": 1.0e-6},
        "metrics": {"ACC": {}, "EER": {}},
        "eval_every_epochs": 1,
        "device": "cuda"
    }
}

# --- Generator Loop ---
count = 0
for frontend in FRONTENDS:
    for backend in BACKENDS:
        for seed in SEEDS:
            for aug in AUGMENTATIONS:
                cfg = copy.deepcopy(BASE_CONFIG)
                
                # Set Name
                f_name = frontend["type"]
                b_name = backend["type"]
                aug_name = aug["name"]
                exp_name = f"PartialSpoof_{f_name}_{b_name}_{aug_name}_seed{seed}"
                
                cfg["exp_name"] = exp_name
                cfg["seed"] = seed
                
                # Set Data Augmentation
                cfg["data"]["train"]["augment_transform"] = aug["augment_transform"]
                
                # Set Model
                cfg["model"]["frontend"] = frontend
                
                # Adjust Backend Config
                b_cfg = copy.deepcopy(backend)
                
                cfg["model"]["backend"] = b_cfg
                
                if b_name == "AASIST":
                     cfg["model"]["loss"][0]["embedding_dim"] = 160
                elif b_name == "Nes2Net":
                     cfg["model"]["loss"][0]["embedding_dim"] = 1024
                else:
                     # Reset to 128 for others (MLP, TCM)
                     cfg["model"]["loss"][0]["embedding_dim"] = 128
                
                # Saving
                filename = f"{exp_name}.yaml"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, "w") as f:
                    yaml.dump(cfg, f, sort_keys=False, default_flow_style=None)
                
                count += 1

print(f"Generated {count} configuration files in {OUTPUT_DIR}")
