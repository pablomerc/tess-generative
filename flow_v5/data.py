import torch
from flow_v5 import config as cfg


def make_triplet_creator(dataset_type: str = None):
    """Factory for TripletCreator using v5 config defaults.

    Imports the canonical TripletCreator from the shared module.
    """
    from triplet_creation import TripletCreator
    ds = dataset_type or cfg.DATASET_TYPE
    return TripletCreator(dataset_type=ds)

def make_multi_triplet_creator(dataset_type: str = None):
    """Factory for TripletCreator with multi-sample batching enabled."""
    from triplet_creation import TripletCreator
    ds = dataset_type or cfg.DATASET_TYPE
    return TripletCreator(dataset_type=ds)
