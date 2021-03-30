import os
from pathlib import Path


def remove_pt_and_bin(ds):
    for d in ds:
        pts = d.rglob('*.pt')
        bins = d.rglob('*.bin')
        for pt in pts:
            if os.path.exists(pt):
                os.remove(pt)
        for bin in bins:
            if os.path.exists(bin):
                os.remove(bin)

