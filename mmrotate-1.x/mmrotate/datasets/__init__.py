# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403
from .dota_pond import PONDDOTADataset
from .dota_tower import TOWERDOTADataset
from .dota_ship import SHIPDOTADataset
from .dota_plane import PLANEDOTADataset
from .dota_tank import TANKDOTADataset
from .dota_bridge import BRIDGEDOTADataset
from .dota_sarplane import SARPLANEDOTADataset

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'PONDDOTADataset', 'TOWERDOTADataset', 'SHIPDOTADataset', 
    'PLANEDOTADataset', 'TANKDOTADataset', 'BRIDGEDOTADataset', 'SARPLANEDOTADataset'
]
