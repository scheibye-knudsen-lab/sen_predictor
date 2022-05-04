import csv
import numpy as np
import os


def prepare_fibrosen_bi(keys=None, path=None):
    BUCKETS = ["ctr", "sen"]

    def class_converter(y, key):
        if "oung" in key:
            return "ctr"
        elif "eplicative" in key:
            return "sen"
        elif "IR" in key and "induced" in key:
            return "sen"
        else:
            return None

    return BUCKETS, None, class_converter


def prepare_folder(keys=None, path=None):
    def class_converter(y, key):
        pos = key.rfind("/")
        if pos < 0:
            return key
        else:
            return key[0:pos]

    buckets = {}

    for key in keys:
        key = class_converter(None, key)
        buckets[key] = 1

    return list(buckets.keys()), None, class_converter
