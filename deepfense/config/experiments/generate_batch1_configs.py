import os
import yaml
import copy

# --- Constants ---
OUTPUT_DIR = "deepfense/config/experiments/batch1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base Paths
TRAIN_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/train.parquet"
VAL_PARQUET = "/netscratch/yelkheir/DeepFense/DeepFense/test.parquet"

# Frontend Checkpoints (Placeholders based on common paths, user can update)
CKPT_WAV2VEC2 = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/models/pretrained/xlsr2_300m.pt"
CKPT_HUBERT = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/models/pretrained/hubert_base_ls960.pt"
CKPT_WAVLM = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/models/pretrained/WavLM-Base+.pt"

# --- Parameters ---
FRONTENDS = [
    {"type": "wav2vec2", "args": {"ckpt_path": CKPT_WAV2VEC2, "freeze": False, "source": "fairseq"}},
    {"type": "hubert", "args": {"ckpt_path": CKPT_HUBERT, "freeze": False, "source": "fairseq"}},
    {"type": "wavlm", "args": {"ckpt_path": CKPT_WAVLM, "freeze": False, "source": "unil"}}
]

BACKENDS = [
    {"type": "AASIST"},
    {"type": "MLP", "args": {"input_dim": 1024, "projection": [128], "pooling_type": "mean"}},
    {"type": "Nes2Net"},
    {"type": "TCM", "args": {"emb_size": 128, "heads": 4, "num_encoders": 1}}
]

SEEDS = [2, 42, 240]

AUGMENTATIONS = [
    {"name": "NoAug", "augment_transform": []},
    {"name": "Concat_RawBoost_RIR", "augment_transform": [
        {
            "type": "augmentation_pipeline",
            "mode": "sequential", 
            "execution": "independent",
            "p": 1.0, 
            "concat_original": True,
            "transforms": [
                {"type": "rawboost", "noise_ratio": 1.0, "algo": 3},
                {"type": "rir", "noise_ratio": 1.0, "csv_file": "rirs.csv"} 
            ]
        }
    ]}
]

# --- Template ---
BASE_CONFIG = {
    "exp_name": "",
    "output_dir": "./outputs/batch1/",
    "seed": 42,
    "data": {
        "sampling_rate": 16000,
        "label_map": {"bonafide": 1, "spoof": 0},
        "train": {
            "dataset_type": "StandardDataset",
            "parquet_files": [TRAIN_PARQUET],
            "dataset_names": ["ASVSpoof19_Train"],
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "base_transform": [
                {"type": "pad", "max_len": 64000, "random_pad": True, "pad_type": "repeat"}
            ],
            "augment_transform": [] 
        },
        "val": {
            "dataset_type": "StandardDataset",
            "parquet_files": [VAL_PARQUET],
            "dataset_names": ["ASVSpoof19_Val"],
            "batch_size": 32,
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
                exp_name = f"{f_name}_{b_name}_{aug_name}_seed{seed}"
                
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
                     # Reset to 128 for others (MLP, TCM, Nes2Net)
                     cfg["model"]["loss"][0]["embedding_dim"] = 128
                
                # Saving
                filename = f"{exp_name}.yaml"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, "w") as f:
                    yaml.dump(cfg, f, sort_keys=False, default_flow_style=None)
                
                count += 1

print(f"Generated {count} configuration files in {OUTPUT_DIR}")
