import os
import yaml
import copy

# --- Constants ---
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/experiments/FakeMusicCaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base Paths
TRAIN_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/FakeMusicCaps/fakemusiccaps_train.parquet"
DEV_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/FakeMusicCaps/fakemusiccaps_dev.parquet"

# --- Parameters ---
# Only EAT and MERT frontends with huggingface source
FRONTENDS = [
    {"type": "eat", "args": {"source": "huggingface"}, "target_sr": 16000},
    {"type": "mert", "args": {"source": "huggingface"}, "target_sr": 24000}
]

BACKENDS = [
    {"type": "AASIST"},
    {"type": "MLP", "args": {"input_dim": 1024, "projection": [128], "pooling_type": "mean"}},
    {"type": "Nes2Net"},
    {"type": "TCM", "args": {"emb_size": 128, "heads": 4, "num_encoders": 1}}
]

SEEDS = [2, 42, 240]

AUGMENTATIONS = [
    {"name": "NoAug", "augment_transform": []}
]

# --- Template ---
BASE_CONFIG = {
    "exp_name": "",
    "output_dir": "./outputs/FakeMusicCaps/",
    "seed": 2,
    "data": {
        "target_sr": 16000,  # Will be adjusted based on frontend
        "label_map": {"bonafide": 1, "spoof": 0},
        "train": {
            "dataset_type": "StandardDataset",
            "parquet_files": [TRAIN_PARQUET],
            "dataset_names": ["FakeMusicCaps_Train"],
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 4,
            "base_transform": [
                {"type": "pad", "max_len": 64000, "random_pad": True, "pad_type": "repeat"}
            ],
            "augment_transform": [] 
        },
        "val": {
            "dataset_type": "StandardDataset",
            "parquet_files": [DEV_PARQUET],
            "dataset_names": ["FakeMusicCaps_Dev"],
            "batch_size": 16,
            "shuffle": False,
            "num_workers": 4,
            "base_transform": [
                {"type": "pad", "max_len": 64000, "pad_type": "repeat"}
            ]
        }
    },
    "model": {
        "type": "StandardDetector",
        "frontend": {},
        "backend": {},
        "loss": [
            {
                "type": "CrossEntropy",
                "embedding_dim": 128,  # Default, adjusted below if backend changes
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
                exp_name = f"FakeMusicCaps_{f_name}_{b_name}_{aug_name}_seed{seed}"
                
                cfg["exp_name"] = exp_name
                cfg["seed"] = seed
                
                # Set target sample rate based on frontend (MERT uses 24kHz, EAT uses 16kHz)
                target_sr = frontend["target_sr"]
                cfg["data"]["target_sr"] = target_sr
                
                # Adjust max_len based on sample rate (4 seconds at target_sr)
                # 16kHz: 64000 samples = 4s
                # 24kHz: 96000 samples = 4s
                max_len = target_sr * 4
                cfg["data"]["train"]["base_transform"][0]["max_len"] = max_len
                cfg["data"]["val"]["base_transform"][0]["max_len"] = max_len
                
                # Set Data Augmentation
                cfg["data"]["train"]["augment_transform"] = aug["augment_transform"]
                
                # Set Model
                cfg["model"]["frontend"] = {"type": frontend["type"], "args": frontend["args"]}
                
                # Adjust Backend Config
                b_cfg = copy.deepcopy(backend)
                cfg["model"]["backend"] = b_cfg
                
                # Set embedding_dim based on backend
                if b_name == "AASIST":
                    cfg["model"]["loss"][0]["embedding_dim"] = 160
                elif b_name == "Nes2Net":
                    cfg["model"]["loss"][0]["embedding_dim"] = 1024  # Hidden size 1024
                elif b_name == "MLP":
                    cfg["model"]["loss"][0]["embedding_dim"] = 128
                else:  # TCM
                    cfg["model"]["loss"][0]["embedding_dim"] = 128
                
                # Saving
                filename = f"{exp_name}.yaml"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, "w") as f:
                    yaml.dump(cfg, f, sort_keys=False, default_flow_style=None)
                
                count += 1

print(f"Generated {count} configuration files in {OUTPUT_DIR}")
