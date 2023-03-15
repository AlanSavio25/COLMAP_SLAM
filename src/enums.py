#Enums for choice of feature extractors and matchers available
from enum import Enum
Extractors = Enum('Extractors', 'SuperPoint ORB')
Matchers = Enum('Matchers', 'SuperGlue OrbHamming')
