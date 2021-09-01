import re
from collections import OrderedDict
from typing import Dict, List

from loguru import logger

from torch import nn

compiled_regex = re.compile("basemodel\\.*")
print(compiled_regex.search("basemodel.layer5.1.bn"))