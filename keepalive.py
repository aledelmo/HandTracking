#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""
from open_gopro.wifi.adapters import Wireless
import json
import requests
import logging
wifi = Wireless()
wifi.connect("GP25875499", "skate6167")


GOPRO_BASE_URL = "10.5.5.100:8554"
url = GOPRO_BASE_URL + f"/gopro/version"

response = requests.get(url)
response.raise_for_status()

print(f"Response: {json.dumps(response.json(), indent=4)}")