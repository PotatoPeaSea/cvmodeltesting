# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
SAM2 Demo Script following the Qualcomm AI Hub Getting Started Guide.
Source: https://github.com/qualcomm/ai-hub-models/tree/v0.53.1/src/qai_hub_models/models/sam2

Prerequisites Installation:
    pip install "qai-hub-models[sam2]" git+https://github.com/facebookresearch/sam2.git@2b90b9f

Usage:
    python sam2_demo.py
    python sam2_demo.py --image <path_or_url> --point-coordinates "x1,y1;x2,y2"
"""

from qai_hub_models.models._shared.sam2.demo import sam2_demo_main
from qai_hub_models.models.sam2.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM2,
    TINY_MODEL_TYPE,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "truck.jpg"
)


# Run SAM end-to-end model on given image.
# The demo will output image with segmentation mask applied for input points
def main(is_test: bool = False) -> None:
    sam2_demo_main(
        model_cls=SAM2,
        model_id=MODEL_ID,
        default_image=IMAGE_ADDRESS,
        default_model_type=TINY_MODEL_TYPE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
