# Datasets

## StandardDataset

`deepfense.data.detection_dataset.StandardDataset`

The default dataset class for reading audio metadata from Parquet files.

**Configuration Args**:
*   `parquet_files` (List[str]): List of paths to parquet files.
*   `label_map` (Dict): Mapping from string label in parquet to integer (e.g., `{"bonafide": 1, "spoof": 0}`).
*   `target_sr` (int): Target sampling rate. Audio will be resampled on load.
*   `base_transform` (List): Transforms applied *always* (e.g., padding/trimming).
*   `augment_transform` (List): Transforms applied *stochastically* (e.g., noise, RIR).
*   `max_per_class` (int, optional): Limit samples per class (useful for debugging or balancing).

**Returns**:
Dictionary containing:
*   `x`: Audio tensor `[T]`.
*   `label`: Label tensor `[1]`.
*   `dataset_name`: Source dataset name string.
*   `ID`: Unique identifier.

