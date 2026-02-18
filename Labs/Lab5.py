import streamlit as st
import requests
import json
from openai import OpenAI

st.set_page_config(page_title="What to Wear Bot", page_icon="🌤️")
st.title("🌤️ Fashion Bot")
st.write("Enter a city and I'll tell you what to wear and suggest outdoor activities!")

# Get API key from secrets
weather_api_key = st.secrets.get("OPENWEATHERMAP_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Get Weather function
def get_current_weather(location, units="imperial"):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={weather_api_key}&units={units}"
    )
    response = requests.get(url)
    if response.status_code == 401:
        raise Exception("Authentication failed: Invalid API key (401 Unauthorized)")
    if response.status_code == 404:
        msg = response.json().get("message", "City not found")
        raise Exception(f"404 error: {msg}")

    data = response.json()
    return {
        "location":    location,
        "temperature": round(data["main"]["temp"], 2),
        "feels_like":  round(data["main"]["feels_like"], 2),
        "temp_min":    round(data["main"]["temp_min"], 2),
        "temp_max":    round(data["main"]["temp_max"], 2),
        "humidity":    round(data["main"]["humidity"], 2),
        "description": data["weather"][0]["description"],
        "wind_speed":  round(data["wind"]["speed"], 2),
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get the current weather for a given location. "
                "Returns temperature (°F), feels-like temp, min/max temp, "
                "humidity, weather description, and wind speed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City name in the format 'City, State, Country' "
                            "e.g. 'Syracuse, NY, US' or 'Lima, Peru'. "
                            "Default to 'Syracuse, NY, US' if no location is provided."
                        ),
                    }
                },
                "required": ["location"],
            },
        },
    }
]

def get_weather_advice(user_input: str) -> str:
    system_msg = (
        "You are a helpful weather-based fashion and activity advisor. "
        "When given weather data, provide friendly, specific suggestions for appropriate clothing to wear today and outdoor activities suited to the current conditions."
        "If no city is mentioned, ask for them to provide a city."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_input},
    ]

    # First API call with tool available
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_msg = response.choices[0].message
    tool_calls   = response_msg.tool_calls

    # If tool was invoked, run it and call the API again
    if tool_calls:
        messages.append(response_msg)  # append assistant's tool-call request
        tool_results = []
        for tc in tool_calls:
            args     = json.loads(tc.function.arguments)
            location = args.get("location", "Syracuse, NY, US")

            try:
                weather_data = get_current_weather(location)
                tool_result  = json.dumps(weather_data)
                st.session_state.api_calls.append({
                    "success":     True,
                    "city":        location,
                    "temp":        weather_data["temperature"],
                    "feels_like":  weather_data["feels_like"],
                    "description": weather_data["description"],
                    "humidity":    weather_data["humidity"],
                })
            except Exception as e:
                tool_result = json.dumps({"error": str(e)})
                st.session_state.api_calls.append({
                    "success": False,
                    "city":    location,
                    "error":   str(e),
                })

            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         tc.function.name,
                "content":      tool_result,
            })

        messages.extend(tool_results)


        # Second call: give the model the weather data so it can advise
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return final_response.choices[0].message.content

    # No tool called – return the model's direct response
    return response_msg.content

# Sidebar 
if "api_calls" not in st.session_state:
    st.session_state.api_calls = []

with st.sidebar:
    st.header("🌐 OpenWeatherMap API Status")

    if st.session_state.api_calls:
        for i, call in enumerate(reversed(st.session_state.api_calls), 1):
            with st.expander(f"Call #{len(st.session_state.api_calls) - i + 1} — {call['city']}"):
                if call["success"]:
                    st.success("✅ API Called Successfully")
                    st.write(f"**City:** {call['city']}")
                    st.write(f"**Temp:** {call['temp']}°F (feels like {call['feels_like']}°F)")
                    st.write(f"**Conditions:** {call['description'].title()}")
                    st.write(f"**Humidity:** {call['humidity']}%")
                else:
                    st.error(f"❌ API Call Failed")
                    st.write(f"**City:** {call['city']}")
                    st.write(f"**Error:** {call['error']}")
        if st.button("Clear History"):
            st.session_state.api_calls = []
            st.rerun()
    else:
        st.info("No API calls made yet. Enter a city and click **Get Advice** to begin.")

# Main Interface
city_input = st.text_input(
    "Enter a city:",
    placeholder="e.g. Syracuse, NY, US  |  Tokyo, Japan  |  Lima, Peru",
)

if st.button("Get Advice"):
    if not city_input.strip():
        query = "What should I wear today? I'm in Syracuse, NY."
    else:
        query = f"What should I wear today and what outdoor activities are good in {city_input}?"

    with st.spinner("Fetching weather and generating advice..."):
        try:
            advice = get_weather_advice(query)
            st.success("Here's your advice!")
            st.write(advice)
        except Exception as e:
            st.error(f"Something went wrong: {e}")