from enum import Enum


class Methods(Enum):
    LTC_LNC = "ltc-lnc"
    LTN_LNN = "ltn-lnn"


class Fields(Enum):
    TITLE = "title"
    TEXT = "text"


FIELDS = [field.value for field in list(Fields)]
