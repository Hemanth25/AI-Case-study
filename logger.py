# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:04:14 2025

@author: hemanthn
"""

import logging
import os

def setup_logger(name: str = "chatbot_logger", log_file: str = "chatbot_log.log"):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(name)
    return logger