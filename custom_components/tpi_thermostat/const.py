"""Constants for the TPI Thermostat helper."""

from homeassistant.components.climate import (
    PRESET_ACTIVITY,
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_ECO,
    PRESET_HOME,
    PRESET_SLEEP,
)
from homeassistant.const import Platform

DOMAIN = "tpi_thermostat"

PLATFORMS = [Platform.CLIMATE]

CONF_AC_MODE = "ac_mode"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HEATER = "heater"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_PRESETS = {
    p: f"{p}_temp"
    for p in (
        PRESET_AWAY,
        PRESET_COMFORT,
        PRESET_ECO,
        PRESET_HOME,
        PRESET_SLEEP,
        PRESET_ACTIVITY,
    )
}
CONF_SENSOR = "target_sensor"
CONF_PROPORTIONAL_BAND = "proportional_band"
CONF_CYCLE_PERIOD = "cycle_period"
DEFAULT_TOLERANCE = 0.3
DEFAULT_PROPORTIONAL_BAND = 2
DEFAULT_CYCLE_PERIOD = 300

THERMOSTAT_STATE_1 = "state_1"
THERMOSTAT_STATE_2 = "state_2"
THERMOSTAT_STATE_3 = "state_3"
THERMOSTAT_STATE_TPI = "tpi"

ATTR_THERMOSTAT_STATE = "thermostat_state"
ATTR_SLOPE = "slope"
ATTR_KP = "kp"
ATTR_TI = "ti"

TREND_UP = "up"
TREND_DOWN = "down"
ZERO_IN_KELVIN = 273.0
DEFAULT_TPI_INTERVAL = 60
SLOPE_TABLE = {
    0.0001: {"kp": 1.086, "ti": 1190},
    0.0004: {"kp": 1.046, "ti": 1160},
    0.0007: {"kp": 1.006, "ti": 1130},
    0.0010: {"kp": 0.966, "ti": 1100},
    0.0013: {"kp": 0.926, "ti": 1070},
    0.0016: {"kp": 0.886, "ti": 1040},
    0.0019: {"kp": 0.846, "ti": 1010},
    0.0022: {"kp": 0.806, "ti": 980},
    0.0025: {"kp": 0.766, "ti": 950},
    0.0028: {"kp": 0.726, "ti": 920},
    0.0031: {"kp": 0.686, "ti": 890},
    0.0034: {"kp": 0.646, "ti": 860},
    0.0037: {"kp": 0.606, "ti": 830},
    0.0040: {"kp": 0.566, "ti": 800},
    0.0043: {"kp": 0.526, "ti": 770},
    0.0046: {"kp": 0.486, "ti": 740},
    0.0049: {"kp": 0.446, "ti": 710},
    0.0052: {"kp": 0.406, "ti": 680},
    0.0055: {"kp": 0.366, "ti": 650},
    0.0058: {"kp": 0.326, "ti": 620},
    0.0061: {"kp": 0.286, "ti": 590},
    0.0064: {"kp": 0.246, "ti": 560},
    0.0067: {"kp": 0.206, "ti": 530},
    0.0070: {"kp": 0.166, "ti": 500},
    0.0073: {"kp": 0.126, "ti": 470},
    0.0076: {"kp": 0.086, "ti": 440},
    0.0079: {"kp": 0.046, "ti": 410},
    0.0082: {"kp": 0.006, "ti": 380},
    0.0085: {"kp": -0.034, "ti": 350},
    0.0088: {"kp": -0.074, "ti": 320},
    0.0091: {"kp": -0.114, "ti": 290},
    0.0094: {"kp": -0.154, "ti": 260},
    0.0097: {"kp": -0.194, "ti": 230},
}
