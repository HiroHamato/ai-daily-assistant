from __future__ import annotations

from typing import Optional
import httpx


class WeatherError(Exception):
    pass


async def _geocode_city(city: str) -> tuple[float, float, str]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "ru",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            raise WeatherError(f"Город не найден: {city}")
        r = results[0]
        return float(r["latitude"]), float(r["longitude"]), r.get("name") or city


async def _fetch_weather(lat: float, lon: float, *, units: str = "metric") -> dict:
    temp_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
        "temperature_unit": temp_unit,
        "wind_speed_unit": wind_unit,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _format_weather(city_name: str, data: dict, *, units: str) -> str:
    cw = data.get("current_weather") or {}
    daily = (data.get("daily") or {})
    t = cw.get("temperature")
    wind = cw.get("windspeed")
    unit = "°C" if units == "metric" else "°F"
    wind_unit = "km/h" if units == "metric" else "mph"

    tmax = None
    tmin = None
    if daily:
        arr_max = daily.get("temperature_2m_max") or []
        arr_min = daily.get("temperature_2m_min") or []
        if arr_max:
            tmax = arr_max[0]
        if arr_min:
            tmin = arr_min[0]

    parts = [f"Погода в {city_name}:"]
    if t is not None:
        parts.append(f"Сейчас: {t}{unit}")
    if wind is not None:
        parts.append(f"Ветер: {wind} {wind_unit}")
    if tmax is not None and tmin is not None:
        parts.append(f"День: {tmin}{unit}…{tmax}{unit}")
    return "\n".join(parts)


async def get_weather(city: str, *, units: str = "metric") -> str:
    lat, lon, normalized = await _geocode_city(city)
    data = await _fetch_weather(lat, lon, units=units)
    return _format_weather(normalized, data, units=units)
