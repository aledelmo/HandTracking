#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

from goprocam import GoProCamera

gopro = GoProCamera.GoPro()
gopro.stream("udp://127.0.0.1:10000")
