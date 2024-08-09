
datasets = {
    "visdrone": {},
    "sds": {
        "mean": [0.4263, 0.4856, 0.4507],
        "std": [0.1638, 0.1515, 0.1755],
        "num_classes": 6,
        "data_root": "",
        "ann_file_train": "",
        "ann_file_val": "",
        "is_sliced": True,
    },
    "sw": {
        "mean": [0.2761, 0.4251, 0.5644],
        "std": [0.2060, 0.1864, 0.2218],
        "num_classes": 1,
        "data_root": "",
        "ann_file_train": "",
        "ann_file_val": "",
        "is_sliced": False,
    }
}

slicing_configs = {
    "15bg_center": {
        "tile_sz": 512,
        "overlap": 0.25,
        "min_area_ratio": 0.95,
        "use_cache": True,
        "target_gsd": None,
        "amount_bg": 0.15,
        "full_image": False
    },
    "no_bg_center":  {
        "tile_sz": 512,
        "overlap": 0.25,
        "min_area_ratio": 0.95,
        "use_cache": True,
        "target_gsd": None,
        "amount_bg": 0,
        "full_image": False
    },
    "50bg_center":  {
        "tile_sz": 512,
        "overlap": 0.25,
        "min_area_ratio": 0.95,
        "use_cache": True,
        "target_gsd": None,
        "amount_bg": 50,
        "full_image": False
    },
    "15bg_full":  {
        "tile_sz": 512,
        "overlap": 0.25,
        "min_area_ratio": 0.1,
        "use_cache": True,
        "target_gsd": None,
        "amount_bg": 15,
        "full_image": False
    },
}





configs = [
    {
        "batch_size": 64,
        "input_shape": (3, 512, 512),
        "backend": "mobileone",
        "bn_momentum": 0.9,
        "epochs": 400,
        "pos_weight": 1,
        "lr": 0.001,
        "loss_type": "dist_bce",
        "attention": None,
        "slicing_config": None,
        "seed": 42
    },
    {
        "batch_size": 64,
        "input_shape": (3, 512, 512),
        "backend": "mobilenet",
        "bn_momentum": 0.9,
        "epochs": 400,
        "pos_weight": 1,
        "lr": 0.001,
        "loss_type": "dist_bce",
        "attention": None,
        "slicing_config": None,
        "seed": 42
    },
]