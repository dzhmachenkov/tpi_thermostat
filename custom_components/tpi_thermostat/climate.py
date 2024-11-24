"""Adds support for Time Proportional and Integral thermostat units."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
import math
from typing import Any

import voluptuous as vol

from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA,
    PRESET_NONE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    DOMAIN as HOMEASSISTANT_DOMAIN,
    CoreState,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, VolDictType

from .const import (
    ATTR_KP,
    ATTR_SLOPE,
    ATTR_TI,
    CONF_AC_MODE,
    CONF_COLD_TOLERANCE,
    CONF_CYCLE_PERIOD,
    CONF_HEATER,
    CONF_HOT_TOLERANCE,
    CONF_PRESETS,
    CONF_PROPORTIONAL_BAND,
    CONF_SENSOR,
    DEFAULT_CYCLE_PERIOD,
    DEFAULT_PROPORTIONAL_BAND,
    DEFAULT_TOLERANCE,
    DOMAIN,
    PLATFORMS,
    SLOPE_TABLE,
    ZERO_IN_KELVIN,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Time Proportional and Integral Thermostat"

CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_PRECISION = "precision"
CONF_TARGET_TEMP = "target_temp"
CONF_TEMP_STEP = "target_temp_step"

PRESETS_SCHEMA: VolDictType = {
    vol.Optional(v): vol.Coerce(float) for v in CONF_PRESETS.values()
}

PLATFORM_SCHEMA_COMMON = vol.Schema(
    {
        vol.Required(CONF_HEATER): cv.entity_id,
        vol.Required(CONF_SENSOR): cv.entity_id,
        vol.Optional(CONF_AC_MODE): cv.boolean,
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(
            CONF_PROPORTIONAL_BAND, default=DEFAULT_PROPORTIONAL_BAND
        ): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVACMode.COOL, HVACMode.HEAT, HVACMode.OFF]
        ),
        vol.Optional(CONF_PRECISION): vol.All(
            vol.Coerce(float),
            vol.In([PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]),
        ),
        vol.Optional(CONF_TEMP_STEP): vol.All(
            vol.In([PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE])
        ),
        vol.Required(
            CONF_PROPORTIONAL_BAND, default=DEFAULT_PROPORTIONAL_BAND
        ): vol.Coerce(float),
        vol.Required(CONF_CYCLE_PERIOD, default=DEFAULT_CYCLE_PERIOD): vol.Coerce(int),
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        **PRESETS_SCHEMA,
    }
)

PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend(PLATFORM_SCHEMA_COMMON.schema)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Initialize config entry."""
    await _async_setup_config(
        hass,
        PLATFORM_SCHEMA_COMMON(dict(config_entry.options)),
        config_entry.entry_id,
        async_add_entities,
    )


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Time Proportional and Integral thermostat platform."""

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    await _async_setup_config(
        hass, config, config.get(CONF_UNIQUE_ID), async_add_entities
    )


async def _async_setup_config(
    hass: HomeAssistant,
    config: Mapping[str, Any],
    unique_id: str | None,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Time Proportional and Integral thermostat platform."""

    name: str = config[CONF_NAME]
    heater_entity_id: str = config[CONF_HEATER]
    sensor_entity_id: str = config[CONF_SENSOR]
    min_temp: float | None = config.get(CONF_MIN_TEMP)
    max_temp: float | None = config.get(CONF_MAX_TEMP)
    target_temp: float | None = config.get(CONF_TARGET_TEMP)
    ac_mode: bool | None = config.get(CONF_AC_MODE)
    cold_tolerance: float = config[CONF_COLD_TOLERANCE]
    hot_tolerance: float = config[CONF_HOT_TOLERANCE]
    proportional_band: float = config[CONF_PROPORTIONAL_BAND]
    initial_hvac_mode: HVACMode | None = config.get(CONF_INITIAL_HVAC_MODE)
    presets: dict[str, float] = {
        key: config[value] for key, value in CONF_PRESETS.items() if value in config
    }
    precision: float | None = config.get(CONF_PRECISION)
    target_temperature_step: float | None = config.get(CONF_TEMP_STEP)
    unit = hass.config.units.temperature_unit
    cycle_period: int = config[CONF_CYCLE_PERIOD]

    async_add_entities(
        [
            TPIThermostat(
                hass,
                name,
                heater_entity_id,
                sensor_entity_id,
                cycle_period,
                min_temp,
                max_temp,
                target_temp,
                ac_mode,
                cold_tolerance,
                hot_tolerance,
                proportional_band,
                initial_hvac_mode,
                presets,
                precision,
                target_temperature_step,
                unit,
                unique_id,
            )
        ]
    )


class TPIThermostat(ClimateEntity, RestoreEntity):
    """Representation of a Time Proportional and Integral Thermostat device."""

    _attr_should_poll = False
    _enable_turn_on_off_backwards_compatibility = False

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        heater_entity_id: str,
        sensor_entity_id: str,
        cycle_period: int,
        min_temp: float | None,
        max_temp: float | None,
        target_temp: float | None,
        ac_mode: bool | None,
        cold_tolerance: float,
        hot_tolerance: float,
        proportional_band: float,
        initial_hvac_mode: HVACMode | None,
        presets: dict[str, float],
        precision: float | None,
        target_temperature_step: float | None,
        unit: UnitOfTemperature,
        unique_id: str | None,
    ) -> None:
        """Initialize the thermostat."""
        self._attr_name = name
        self.heater_entity_id = heater_entity_id
        self.sensor_entity_id = sensor_entity_id
        self._attr_device_info = async_device_info_to_link_from_entity(
            hass,
            heater_entity_id,
        )
        self.ac_mode = ac_mode
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._proportional_band = proportional_band
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp or next(iter(presets.values()), None)
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        if self.ac_mode:
            self._attr_hvac_modes = [HVACMode.COOL, HVACMode.OFF]
        else:
            self._attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
        self._active = False
        self._cur_temp: float | None = None
        self._cur_temp_time: datetime | None = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._attr_supported_features = (
            ClimateEntityFeature.TARGET_TEMPERATURE
            | ClimateEntityFeature.TURN_OFF
            | ClimateEntityFeature.TURN_ON
        )
        if len(presets):
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE, *presets.keys()]
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._presets = presets
        self._cycle_period = cycle_period
        self._slope_start: tuple[float, datetime] | None = None
        self._slope_end: tuple[float, datetime] | None = None
        self._prev_cur_temp_in_kelvin: float | None = None
        self._prev_cur_temp_time: datetime | None = None
        self._attr_slope: float | None = None
        self._attr_kp: float | None = None
        self._attr_ti: float | None = None
        self._tpi_time_interval: CALLBACK_TYPE | None = None
        self._tpi_error: float | None = None
        self._last_pulse_time: datetime | None = None
        self._last_interval_time: datetime | None = None
        self._duty: float = 0.0
        self._signal: bool = False
        self._pwm_interval: CALLBACK_TYPE | None = None

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return entity specific state attributes."""
        data = {
            ATTR_SLOPE: self._attr_slope,
            ATTR_KP: self._attr_kp,
            ATTR_TI: self._attr_ti,
        }

        extra_state_attributes = super().extra_state_attributes

        if extra_state_attributes is not None:
            data.update(extra_state_attributes)

        return data

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.sensor_entity_id], self._async_sensor_changed
            )
        )
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.heater_entity_id], self._async_switch_changed
            )
        )

        @callback
        def _async_startup(_: Event | None = None) -> None:
            """Init on startup."""
            sensor_state = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self._async_update_temp(sensor_state)
                self.async_write_ha_state()
            switch_state = self.hass.states.get(self.heater_entity_id)
            if switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self.hass.async_create_task(
                    self._check_switch_initial_state(), eager_start=True
                )

        if self.hass.state is CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Check If we have an old state
        if (old_state := await self.async_get_last_state()) is not None:
            if (
                self._attr_slope is None
                and old_state.attributes.get(ATTR_SLOPE) is not None
            ):
                self._attr_slope = old_state.attributes.get(ATTR_SLOPE)
                if self._attr_slope is not None:
                    self._attr_kp, self._attr_ti = self._get_kd_and_ti(self._attr_slope)
            if self._attr_slope is not None:
                self._pwm_start()
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    if self.ac_mode:
                        self._target_temp = self.max_temp
                    else:
                        self._target_temp = self.min_temp
                    _LOGGER.warning(
                        "Undefined target temperature, falling back to %s",
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if (
                self.preset_modes
                and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes
            ):
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = HVACMode(old_state.state)
        else:
            # No previous state, try and restore defaults
            if self._target_temp is None:
                if self.ac_mode:
                    self._target_temp = self.max_temp
                else:
                    self._target_temp = self.min_temp
            _LOGGER.warning(
                "No previously saved temperature, setting to %s", self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

    @property
    def precision(self) -> float:
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self) -> float:
        """Return the supported step of target temperature."""
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        # if a target_temperature_step is not defined, fallback to equal the precision
        return self.precision

    @property
    def current_temperature(self) -> float | None:
        """Return the sensor temperature."""
        return self._cur_temp

    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE
        if self.ac_mode:
            return HVACAction.COOLING
        return HVACAction.HEATING

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._target_temp

    @property
    def _cur_temp_in_kelvin(self) -> float | None:
        return None if self._cur_temp is None else self._cur_temp + ZERO_IN_KELVIN

    @property
    def _target_temp_in_kelvin(self) -> float | None:
        return None if self._target_temp is None else self._target_temp + ZERO_IN_KELVIN

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.HEAT:
            self._hvac_mode = HVACMode.HEAT
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.COOL:
            self._hvac_mode = HVACMode.COOL
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.OFF:
            self._hvac_mode = HVACMode.OFF
            if self._is_device_active:
                await self._async_heater_turn_off()
        else:
            _LOGGER.error("Unrecognized hvac mode: %s", hvac_mode)
            return
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        await self._async_control_heating(force=True)
        self.async_write_ha_state()

    @property
    def min_temp(self) -> float:
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp

    async def _async_sensor_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle temperature changes."""
        new_state = event.data["new_state"]
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _check_switch_initial_state(self) -> None:
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_device_active:
            _LOGGER.warning(
                (
                    "The climate mode is OFF, but the switch device is ON. Turning off"
                    " device %s"
                ),
                self.heater_entity_id,
            )
            await self._async_heater_turn_off()

    @callback
    def _async_switch_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle heater switch state changes."""
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]
        if new_state is None:
            return
        if old_state is None:
            self.hass.async_create_task(
                self._check_switch_initial_state(), eager_start=True
            )
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, state: State) -> None:
        """Update thermostat with latest state from sensor."""
        try:
            cur_temp = float(state.state)
            if not math.isfinite(cur_temp):
                raise ValueError(  # noqa: TRY301
                    f"Sensor has illegal state {state.state}"
                )
            if self._cur_temp_in_kelvin is not None:
                self._prev_cur_temp_in_kelvin = self._cur_temp_in_kelvin
                self._prev_cur_temp_time = self._cur_temp_time
            self._cur_temp = cur_temp
            self._cur_temp_time = datetime.now()
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)

    async def _async_control_heating(self, force: bool = False) -> None:
        """Check if we need to turn heating on or off."""
        async with self._temp_lock:
            if not self._active and None not in (
                self._cur_temp,
                self._target_temp,
            ):
                self._active = True
                _LOGGER.debug(
                    (
                        "Obtained current and target temperature. "
                        "Time Proportional and Integral Thermostat active. %s, %s"
                    ),
                    self._cur_temp,
                    self._target_temp,
                )

            if not self._active or self._hvac_mode == HVACMode.OFF:
                return

            if force:
                self._pwm_stop()

            assert self._cur_temp is not None and self._target_temp is not None
            if self.ac_mode:
                too_cold = self._target_temp >= self._cur_temp + self._cold_tolerance
                too_hot = self._cur_temp >= self._target_temp + self._hot_tolerance
                if self._is_device_active:
                    if too_cold:
                        _LOGGER.debug("Turning off heater %s", self.heater_entity_id)
                        await self._async_heater_turn_off()
                elif too_hot:
                    _LOGGER.debug("Turning on heater %s", self.heater_entity_id)
                    await self._async_heater_turn_on()
            else:
                assert isinstance(self._cur_temp_in_kelvin, float)
                assert isinstance(self._target_temp_in_kelvin, float)
                too_cold = (
                    self._cur_temp_in_kelvin
                    <= self._target_temp_in_kelvin - self._proportional_band / 2
                )
                too_hot = (
                    self._cur_temp_in_kelvin
                    >= self._target_temp_in_kelvin + self._proportional_band / 2
                )
                _LOGGER.debug(
                    "%s - Control heating: too_cold: %s too_hot: %s prev_temp: %s cur_temp: %s",
                    self.entity_id,
                    too_cold,
                    too_hot,
                    self._prev_cur_temp_in_kelvin,
                    self._cur_temp_in_kelvin,
                )

                # If On/Off state
                if self._attr_slope is None or self._pwm_interval is None:
                    if self._is_device_active:
                        if too_hot:
                            _LOGGER.debug(
                                "Turning off heater %s",
                                self.heater_entity_id,
                            )
                            await self._async_heater_turn_off()
                            if self._slope_start is not None:
                                self._slope_end = (
                                    self._cur_temp_in_kelvin,
                                    datetime.now(),
                                )
                                _LOGGER.debug(
                                    "%s - [slope end] T2: %s (%s)",
                                    self.entity_id,
                                    self._slope_end[0],
                                    self._slope_end[1],
                                )
                                if (
                                    self._slope_start is not None
                                    and self._slope_end is not None
                                ):
                                    self._attr_slope = self._calculate_slope()

                                    if self._attr_slope is not None:
                                        self._attr_kp, self._attr_ti = (
                                            self._get_kd_and_ti(self._attr_slope)
                                        )
                                    else:
                                        self._attr_kp = None
                                        self._attr_ti = None
                                    self.async_write_ha_state()
                    elif too_cold:
                        _LOGGER.debug(
                            "Turning on heater %s",
                            self.heater_entity_id,
                        )
                        await self._async_heater_turn_on()
                        self._slope_start = (
                            self._cur_temp_in_kelvin,
                            datetime.now(),
                        )
                        self._slope_end = None
                        _LOGGER.debug(
                            "%s - [slope start] T1: %s (%s)",
                            self.entity_id,
                            self._slope_start[0],
                            self._slope_start[1],
                        )
                        if self._attr_slope is not None:
                            self._pwm_start()
                else:
                    assert isinstance(self._prev_cur_temp_time, float)

                    if self._is_device_active and not self._signal:
                        _LOGGER.debug(
                            "Turning off heater %s",
                            self.heater_entity_id,
                        )
                        await self._async_heater_turn_off()
                        if self._slope_start is not None:
                            self._slope_end = (
                                self._cur_temp_in_kelvin,
                                datetime.now(),
                            )
                            _LOGGER.debug(
                                "%s - [slope end] T2: %s (%s)",
                                self.entity_id,
                                self._slope_end[0],
                                self._slope_end[1],
                            )
                            if (
                                self._slope_start is not None
                                and self._slope_end is not None
                            ):
                                self._attr_slope = self._calculate_slope()

                                if self._attr_slope is not None:
                                    self._attr_kp, self._attr_ti = self._get_kd_and_ti(
                                        self._attr_slope
                                    )
                                elif self._pwm_interval is not None:
                                    self._pwm_stop()
                                self.async_write_ha_state()
                    elif not self._is_device_active and self._signal:
                        _LOGGER.debug(
                            "Turning on heater %s",
                            self.heater_entity_id,
                        )
                        await self._async_heater_turn_on()
                        self._slope_start = (
                            self._cur_temp_in_kelvin,
                            datetime.now(),
                        )
                        self._slope_end = None
                        _LOGGER.debug(
                            "%s - [slope start] T1: %s (%s)",
                            self.entity_id,
                            self._slope_start[0],
                            self._slope_start[1],
                        )

    @property
    def _is_device_active(self) -> bool | None:
        """If the toggleable device is currently active."""
        if not self.hass.states.get(self.heater_entity_id):
            return None

        return self.hass.states.is_state(self.heater_entity_id, STATE_ON)

    async def _async_heater_turn_on(self) -> None:
        """Turn heater toggleable device on."""
        data = {ATTR_ENTITY_ID: self.heater_entity_id}
        await self.hass.services.async_call(
            HOMEASSISTANT_DOMAIN, SERVICE_TURN_ON, data, context=self._context
        )

    async def _async_heater_turn_off(self) -> None:
        """Turn heater toggleable device off."""
        data = {ATTR_ENTITY_ID: self.heater_entity_id}
        await self.hass.services.async_call(
            HOMEASSISTANT_DOMAIN, SERVICE_TURN_OFF, data, context=self._context
        )

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode."""
        if preset_mode not in (self.preset_modes or []):
            raise ValueError(
                f"Got unsupported preset_mode {preset_mode}. Must be one of"
                f" {self.preset_modes}"
            )
        if preset_mode == self._attr_preset_mode:
            # I don't think we need to call async_write_ha_state if we didn't change the state
            return
        if preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control_heating(force=True)
        else:
            if self._attr_preset_mode == PRESET_NONE:
                self._saved_target_temp = self._target_temp
            self._attr_preset_mode = preset_mode
            self._target_temp = self._presets[preset_mode]
            await self._async_control_heating(force=True)

        self.async_write_ha_state()

    def _calculate_tpi(
        self,
        error: float,
        kp: float,
        ti: float,
        ts: float,
        out_old: float | None = None,
        error_old: float | None = None,
    ) -> tuple[float, float]:
        out = (
            0.5 + kp * error
            if out_old is None or error_old is None
            else out_old + kp * (error - error_old) + (kp * ts * error) / ti
        )
        out = 0 if out < 0 else min(out, 1)
        return (out, error)

    def _get_pwm(self, t: datetime, d: float) -> bool:
        pw = timedelta(self._cycle_period * d)

        assert isinstance(self._last_pulse_time, datetime)

        if self._last_pulse_time < t < self._last_pulse_time + pw:
            return True
        if (
            self._last_pulse_time + pw
            < t
            < self._last_pulse_time + timedelta(seconds=self._cycle_period)
        ):
            return False
        return False

    def _pwm_start(self) -> None:
        if self._pwm_interval is not None:
            self._pwm_stop()
        self._pwm_interval = async_track_time_interval(
            self.hass,
            self._async_pwm_interval_handler,
            timedelta(seconds=self._cycle_period),
        )
        _LOGGER.debug("%s - PWM started", self.entity_id)
        self._pwm_period_handler()

    def _pwm_stop(self) -> None:
        self._signal = False
        self._attr_slope = None
        self._attr_kp = None
        self._attr_ti = None

        if self._pwm_interval is not None:
            self._pwm_interval()
            self._pwm_interval = None
        _LOGGER.debug("%s - PWM stopped", self.entity_id)

    def _calculate_slope(self) -> float | None:
        _LOGGER.debug(
            "%s - Calculate slope: start: %s, end: %s",
            self.entity_id,
            self._slope_start,
            self._slope_end,
        )
        if self._slope_start is None or self._slope_end is None:
            return None
        t = float(self._slope_end[0] - self._slope_start[0])
        d = float((self._slope_end[1] - self._slope_start[1]).seconds)
        return round(t / d, 4) if d != 0.0 else None

    def _get_kd_and_ti(self, slope: float) -> tuple[float, float]:
        delta = 0.00015
        lower = min(SLOPE_TABLE.keys())
        highger = max(SLOPE_TABLE.keys())

        if slope < lower:
            result = (SLOPE_TABLE[lower]["kp"], SLOPE_TABLE[lower]["ti"])
        elif slope > highger:
            result = (SLOPE_TABLE[highger]["kp"], SLOPE_TABLE[highger]["ti"])
        else:
            key = next(x for x in SLOPE_TABLE if x - delta <= slope < x + delta)
            result = (SLOPE_TABLE[key]["kp"], SLOPE_TABLE[key]["ti"])

        _LOGGER.debug(
            "%s - Selected kp: %f, ti: %d for slope: %f",
            self.entity_id,
            result[0],
            result[1],
            slope,
        )
        return result

    async def _async_pwm_interval_handler(self, time: datetime) -> None:
        assert isinstance(self._cur_temp_in_kelvin, float)
        assert isinstance(self._target_temp_in_kelvin, float)
        assert isinstance(self._attr_kp, float)
        assert isinstance(self._attr_ti, float)

        if self._last_pulse_time is None:
            self._last_pulse_time = time
        else:
            delta = (self._last_pulse_time - time).total_seconds()
            if delta >= self._cycle_period:
                self._last_pulse_time = time - timedelta(
                    seconds=delta % self._cycle_period
                )
        signal = self._signal
        ts = (
            0
            if self._last_interval_time is None
            else (time - self._last_interval_time).total_seconds()
        )
        self._last_interval_time = time
        error = (
            self._cur_temp_in_kelvin - self._target_temp_in_kelvin
        ) / self._proportional_band
        self._duty, self._tpi_error = self._calculate_tpi(
            error=error,
            kp=self._attr_kp,
            ti=self._attr_ti,
            ts=ts,
            out_old=self._duty,
            error_old=self._tpi_error,
        )
        self._signal = self._get_pwm(time, self._duty)
        if signal != self._signal:
            await self._async_control_heating()
        _LOGGER.debug(
            "%s - PWM interval last pulse time: %s, duty: %s, sifnal: %s",
            self.entity_id,
            (
                self._last_pulse_time.strftime("%d/%m/%Y %H:%M:%S")
                if self._last_pulse_time is not None
                else None
            ),
            self._duty,
            self._signal,
        )