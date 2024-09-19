from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    data_provided = False
    hierarchical = None
    ignore_hierarchical_value = None
    add_manual_seg_columns = None

    survey_name: str
    trunc_survey_name: str
    sp_tag: bool
    environ: str
    data_uri: str
    columns_uri: str
    preprocessed: str
    weight_column: str
