"""Adds support for Time Proportional and Integral thermostat units."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
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
    HassJob,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
    async_call_later,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, VolDictType

from .const import (
    ATTR_KP,
    ATTR_SLOPE,
    ATTR_TI,
    CONF_AC_MODE,
    CONF_CYCLE_PERIOD,
    CONF_HEATER,
    CONF_PRESETS,
    CONF_PROPORTIONAL_BAND,
    CONF_SENSOR,
    DEFAULT_CYCLE_PERIOD,
    DEFAULT_PROPORTIONAL_BAND,
    DOMAIN,
    PLATFORMS,
    ATTR_THERMOSTAT_STATE,
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

    THERMOSTAT_STATE_1 = "state_1"
    THERMOSTAT_STATE_2 = "state_2"
    THERMOSTAT_STATE_3 = "state_3"
    THERMOSTAT_STATE_TPI = "state_tpi"

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
        self._cold_tolerance = proportional_band / 2
        self._hot_tolerance = proportional_band / 2
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
        self._proportional_band = proportional_band
        self._attr_thermostat_state: str | None = None
        self._slope_start: tuple[float, datetime] | None = None
        self._slope_end: tuple[float, datetime] | None = None
        self._peak_count: int = 0
        self._prev_cur_temp: float | None = None
        self._prev_cur_temp_time: datetime | None = None
        self._attr_slope: float | None = None
        self._attr_kp: float | None = None
        self._attr_ti: float | None = None
        self._tpi_time_interval: CALLBACK_TYPE | None = None
        self._tpi_error: float | None = None
        self._last_pulse_time: datetime | None = None
        self._last_interval_time: datetime | None = None
        self._duty: float = 0.0
        self._signal: bool | None = None
        self._pwm_period: CALLBACK_TYPE | None = None
        self._pwm_duty: CALLBACK_TYPE | None = None
        self._prev_temp_0_1: int | None = None

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return entity specific state attributes."""
        data = {
            ATTR_THERMOSTAT_STATE: self._attr_thermostat_state,
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
            if (
                self._attr_thermostat_state is None
                and old_state.attributes.get(ATTR_THERMOSTAT_STATE) is not None
            ):
                await self._async_to_tpi_state(
                    old_state.attributes.get(ATTR_THERMOSTAT_STATE)
                )

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
                await self._async_to_tpi_state(None)
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
            if self._cur_temp is not None:
                self._prev_cur_temp = self._cur_temp
                self._prev_cur_temp_time = self._cur_temp_time
            self._cur_temp = cur_temp
            self._cur_temp_time = datetime.now(timezone.utc)
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)

    async def _async_control_heating(
        self, time: datetime | None = None, force: bool = False
    ) -> None:
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

            # reset TPI
            if force:
                _LOGGER.debug("%s - Reset TPI.", self.entity_id)
                await self._async_to_tpi_state(None)

            assert self._cur_temp is not None and self._target_temp is not None
            max_temp = self._target_temp + self._hot_tolerance
            min_temp = self._target_temp - self._cold_tolerance
            too_cold = self._cur_temp <= min_temp
            too_hot = self._cur_temp >= max_temp

            _LOGGER.debug(
                "%s - Control heating: min: %s max: %s too_cold: %s too_hot: %s ac_mode: %s target_temp: %s prev_temp: %s cur_temp: %s",
                self.entity_id,
                min_temp,
                max_temp,
                too_cold,
                too_hot,
                self.ac_mode,
                self._target_temp,
                self._prev_cur_temp,
                self._cur_temp,
            )

            # get TPI state
            if self._attr_thermostat_state is None:
                if self._cur_temp < min_temp:
                    await self._async_to_tpi_state(self.THERMOSTAT_STATE_1)
                elif self._cur_temp > max_temp:
                    await self._async_to_tpi_state(self.THERMOSTAT_STATE_2)
                else:
                    await self._async_to_tpi_state(self.THERMOSTAT_STATE_3)
            # else:
            #     if (self._attr_thermostat_state == self.THERMOSTAT_STATE_1 and self._peak_count >= 1) or (self._attr_thermostat_state in [self.THERMOSTAT_STATE_2, self.THERMOSTAT_STATE_3] and self._peak_count >= 2):
            #         _LOGGER.debug("%s - state: %s peak count: %d", self.entity_id, self._attr_thermostat_state, self._peak_count)
            #         self._to_tpi_state(self.THERMOSTAT_STATE_TPI)

            if self.ac_mode:
                if self._is_device_active:
                    if too_cold:
                        _LOGGER.debug("Turning off heater %s", self.heater_entity_id)
                        await self._async_heater_turn_off()
                elif too_hot:
                    _LOGGER.debug("Turning on heater %s", self.heater_entity_id)
                    await self._async_heater_turn_on()
            else:
                # peeks count when on/off heater in not TPI state
                if self._attr_thermostat_state in [
                    self.THERMOSTAT_STATE_1,
                    self.THERMOSTAT_STATE_2,
                    self.THERMOSTAT_STATE_3,
                ]:
                    # turn on/off heater
                    if self._is_device_active and too_hot:
                        _LOGGER.debug(
                            "Turning off heater %s",
                            self.heater_entity_id,
                        )
                        await self._async_heater_turn_off()
                    elif not self._is_device_active and too_cold:
                        _LOGGER.debug(
                            "Turning on heater %s",
                            self.heater_entity_id,
                        )
                        self._peak_count += 1
                        await self._async_heater_turn_on()
                        if (
                            self._attr_thermostat_state
                            in [
                                self.THERMOSTAT_STATE_2,
                                self.THERMOSTAT_STATE_2,
                                self.THERMOSTAT_STATE_3,
                            ]
                            and self._peak_count >= 2
                        ):
                            _LOGGER.debug(
                                "%s - state: %s peak count: %d",
                                self.entity_id,
                                self._attr_thermostat_state,
                                self._peak_count,
                            )
                            await self._async_to_tpi_state(self.THERMOSTAT_STATE_TPI)

                    # get start and end slope
                    if (
                        self._prev_temp_0_1 is not None
                        and self._prev_cur_temp is not None
                        and self._prev_temp_0_1 != self._temp_0_1
                    ):
                        _LOGGER.debug(
                            "%s - Active: %s Prev temp 0 & 1: %s Temp 0 & 1: %s",
                            self.entity_id,
                            self._is_device_active,
                            self._prev_temp_0_1,
                            self._temp_0_1,
                        )

                        if self._prev_cur_temp < self._cur_temp and self._temp_0_1 == 0:
                            self._slope_start = (
                                self._cur_temp,
                                datetime.now(timezone.utc),
                            )
                            self._slope_end = None
                        elif (
                            self._prev_cur_temp < self._cur_temp and self._temp_0_1 == 1
                        ):
                            self._slope_end = (
                                self._cur_temp,
                                datetime.now(timezone.utc),
                            )

                        self._prev_temp_0_1 = self._temp_0_1
                    elif self._prev_temp_0_1 is None:
                        self._prev_temp_0_1 = self._temp_0_1

                    # calculate slope
                    if self._slope_start is not None and self._slope_end is not None:
                        self._attr_slope = self._calculate_slope()
                        self._slope_start = None
                        self._slope_end = None

                        if self._attr_slope is not None:
                            self._attr_kp, self._attr_ti = self._get_kd_and_ti(
                                self._attr_slope
                            )
                            _LOGGER.debug(
                                "%s - Selected kp: %f, ti: %d for slope: %f",
                                self.entity_id,
                                self._attr_kp,
                                self._attr_ti,
                                self._attr_slope,
                            )
                        self.async_write_ha_state()
                else:
                    if self._is_device_active and not self._signal:
                        _LOGGER.debug(
                            "Turning off heater %s",
                            self.heater_entity_id,
                        )
                        self._slope_end = (self._cur_temp, datetime.now(timezone.utc))
                        await self._async_heater_turn_off()

                        # calculate slope
                        if (
                            self._slope_start is not None
                            and self._slope_end is not None
                        ):
                            self._attr_slope = self._calculate_slope()
                            self._slope_start = None
                            self._slope_end = None

                            if self._attr_slope is not None:
                                self._attr_kp, self._attr_ti = self._get_kd_and_ti(
                                    self._attr_slope
                                )
                                _LOGGER.debug(
                                    "%s - Selected kp: %f, ti: %d for slope: %f",
                                    self.entity_id,
                                    self._attr_kp,
                                    self._attr_ti,
                                    self._attr_slope,
                                )
                            self.async_write_ha_state()
                    elif not self._is_device_active and self._signal:
                        _LOGGER.debug(
                            "Turning on heater %s",
                            self.heater_entity_id,
                        )
                        self._slope_start = (self._cur_temp, datetime.now(timezone.utc))
                        self._slope_end = None
                        await self._async_heater_turn_on()

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

    @property
    def _temp_0_1(self):
        return (
            None
            if self._cur_temp is None
            else 0
            if self._target_temp + self._hot_tolerance
            > self._cur_temp
            > self._target_temp - self._cold_tolerance
            else 1
        )

    async def _async_to_tpi_state(self, state: str | None):
        _LOGGER.debug(
            "%s - Change state %s to %s.",
            self.entity_id,
            self._attr_thermostat_state,
            state,
        )

        if self._attr_thermostat_state != state and state in [
            self.THERMOSTAT_STATE_1,
            self.THERMOSTAT_STATE_2,
            self.THERMOSTAT_STATE_3,
            self.THERMOSTAT_STATE_TPI,
            None,
        ]:
            if state is None:
                self._attr_thermostat_state = None
                self._slope_start = None
                self._slope_end = None
                self._attr_slope = None
                self._attr_kp = None
                self._attr_ti = None
                self._peak_count = 0
                self._prev_temp_0_1 = None
            if state == self.THERMOSTAT_STATE_TPI:
                self._slope_start = None
                self._slope_end = None
                self._peak_count = 0
                self._duty = 0.0
                self._tpi_error = None
                await self._async_pwm_start()
            if self._attr_thermostat_state == self.THERMOSTAT_STATE_TPI:
                await self._async_pwm_stop()

            self._attr_thermostat_state = state

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

    def _calculate_slope(self) -> float | None:
        if self._slope_start is None or self._slope_end is None:
            return None
        t = float(self._slope_end[0] - self._slope_start[0])
        d = float((self._slope_end[1] - self._slope_start[1]).seconds)
        slope = round(t / d, 4) if d != 0.0 else None

        _LOGGER.debug(
            "%s - Calculate slope: start: %s, end: %s, slope: %s",
            self.entity_id,
            self._slope_start,
            self._slope_end,
            slope,
        )

        return slope

    def _get_pwm(self, t: datetime, duty: float) -> bool:
        pw = timedelta(seconds=self._cycle_period * duty)

        assert isinstance(self._last_pulse_time, datetime)

        if self._last_pulse_time <= t < self._last_pulse_time + pw:
            return True
        return False

    async def _async_pwm_start(self) -> None:
        @callback
        async def async_pwm_duty_end(time: datetime | None = None):
            signal = self._signal
            self._signal = False

            _LOGGER.debug(
                "%s - PWM duty end: time: %s, last pulse time: %s, duty: %s, signal: %s",
                self.entity_id,
                (
                    self._last_pulse_time.strftime("%d/%m/%Y %H:%M:%S")
                    if self._last_pulse_time is not None
                    else None
                ),
                time.strftime("%d/%m/%Y %H:%M:%S"),
                self._duty,
                self._signal,
            )

            if signal != self._signal:
                async_call_later(
                    self.hass,
                    0,
                    HassJob(self._async_control_heating, cancel_on_shutdown=True),
                )

        @callback
        async def async_pwm_period(time: datetime | None = None):
            self._last_pulse_time = datetime.now(timezone.utc) if time is None else time
            signal = self._signal

            # calculate TPI
            ts = self._cycle_period
            error = (
                (self._cur_temp - self._target_temp) / self._proportional_band
                if self.ac_mode
                else (self._target_temp - self._cur_temp) / self._proportional_band
            )
            self._duty, self._tpi_error = self._calculate_tpi(
                error=error,
                kp=self._attr_kp,
                ti=self._attr_ti,
                ts=ts,
                out_old=self._duty,
                error_old=self._tpi_error,
            )
            _LOGGER.debug(
                "%s - Calculated duty: %f (%f), target temp: % current temp: %s",
                self.entity_id,
                self._duty,
                self._cycle_period * self._duty,
                self._target_temp,
                self._cur_temp,
            )

            if 0 < self._duty < 1:
                self._signal = True
                duty_period = self._cycle_period * self._duty
                self._pwm_duty = async_call_later(
                    self.hass, duty_period, async_pwm_duty_end
                )
                self.async_on_remove(self._pwm_duty)
            else:
                self._signal = False

            _LOGGER.debug(
                "%s - PWM period: last pulse time: %s, duty: %s, signal: %s",
                self.entity_id,
                (
                    self._last_pulse_time.strftime("%d/%m/%Y %H:%M:%S")
                    if self._last_pulse_time is not None
                    else None
                ),
                self._duty,
                self._signal,
            )

            if signal != self._signal:
                async_call_later(
                    self.hass,
                    0,
                    HassJob(
                        self._async_control_heating,
                        "_async_control_heating",
                        cancel_on_shutdown=True,
                    ),
                )

        self._pwm_period = async_track_time_interval(
            self.hass,
            async_pwm_period,
            timedelta(seconds=self._cycle_period),
            cancel_on_shutdown=True,
        )
        self.async_on_remove(self._pwm_period)

        _LOGGER.debug("%s - PWM started", self.entity_id)

        await async_pwm_period()

    async def _async_pwm_stop(self) -> None:
        self._signal = False

        if self._pwm_duty is not None:
            self._pwm_duty()
            self._pwm_duty = None

        if self._pwm_period is not None:
            self._pwm_period()
            self._pwm_period = None
        _LOGGER.debug("%s - PWM stopped", self.entity_id)

    @property
    def _is_pwm_started(self):
        return self._pwm_period is not None or self._pwm_period is not None

    def _get_kd_and_ti(self, slope: float) -> tuple[float, float]:
        slope_start = 10.0
        slope_end = 70.0
        kp_start = 0.966
        kp_step = 0.01333
        ti_start = 1100.0
        ti_step = 10.0
        slope = slope * 10000
        slope = (
            slope_start
            if slope < slope_start
            else slope_end
            if slope > slope_end
            else slope
        )
        kp = kp_start - (slope - slope_start) * kp_step
        ti = ti_start - (slope - slope_start) * ti_step
        return (round(kp, 3), round(ti, 0))
