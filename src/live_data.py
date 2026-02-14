import requests
import json

def get_location():
    """
    Fetches the user's approximate location based on IP address.
    Returns: (latitude, longitude, city)
    """
    try:
        # Use a free IP-geolocation service
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return data['lat'], data['lon'], data['city']
    except Exception as e:
        print(f"Location Error: {e}")
        return 40.7128, -74.0060, "New York (Default)"

def get_real_weather(lat, lon):
    """
    Fetches REAL-TIME weather from Open-Meteo (No API Key needed).
    Returns: Dictionary of weather conditions formatted for your model.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,is_day,precipitation,weather_code,visibility",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph"
        }
        
        response = requests.get(url, params=params)
        data = response.json()['current']
        
        # Mapping Open-Meteo WMO codes to your Model's categories
        # Codes: 0=Clear, 1-3=Cloudy, 50-69=Rain, 70-79=Snow, 45-48=Fog
        wmo_code = data['weather_code']
        weather_condition = "Clear"
        
        if wmo_code in [1, 2, 3]: weather_condition = "Cloudy"
        elif wmo_code in [45, 48]: weather_condition = "Fog"
        elif wmo_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: weather_condition = "Rain"
        elif wmo_code in [71, 73, 75, 77, 85, 86]: weather_condition = "Snow"
        
        # Visibility comes in meters, convert to miles for your model
        vis_miles = data['visibility'] / 1609.34
        
        return {
            "temperature": data['temperature_2m'],
            "visibility": min(vis_miles, 10.0), # Cap at 10 miles
            "condition": weather_condition,
            "is_day": data['is_day']
        }
        
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None