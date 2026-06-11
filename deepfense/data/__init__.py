# data/__init__.py

# Import all dataset definitions so they register themselves

from deepfense.data.detection_dataset import *
from deepfense.data.temporal_dataset import *  # noqa: F401  TemporalSegmentationDataset
