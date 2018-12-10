# -*- coding: utf-8 -*-

"""\
 Analytics App Config
Copyright (c) 2015-2018, MGH Computational Pathology

"""
import os

# Prediction Service app settings
CONCURRENT_USERS = 5
flask_addr = os.environ.get("PS_ADDR")
flask_prt = int(os.environ.get("PS_PRT", 5000))
