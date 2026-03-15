# -*- coding: utf-8 -*-
"""
GOCI-UST21 Unified Restoration Testbed
======================================

GOCI (Remote Sensing Reflectance)와 UST21 (Chlorophyll-a) 모델을
통합하여 테스트할 수 있는 테스트 베드입니다.

Usage:
    python -m testbed.run_testbed --config testbed/configs/testbed.yaml
    python -m testbed.run_testbed --model goci --bands 2 4
    python -m testbed.run_testbed --model ust21 --start 20210101 --end 20210131
"""

__version__ = '1.0.0'
__author__ = 'AY_ust'
