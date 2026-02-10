import requests
from datetime import date, timedelta
import json


SECRET_FILE = "secrets"   # this is where the plaintext API key is stored
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Durango,CO"
#          /[date1]/[date2]?key=YOUR_API_KEY


def _parse_response(res):
    """helper for ForcastClient.get_forecast()"""
    rows = []
    for day in res["days"]:
        for hour in day["hours"]:
            rows.append({
                "date": day["datetime"],
                "time": hour["datetime"],
                **hour
            })
    return rows


"""
This class is responsible for getting the weather forecast from Visual Crossing Weather's API.

    An API key is required. Here is how to set it up:
        1. Create an account on Visual Crossing Weather's Website, and get an API key.
        2. Create a file and set SECRET_FILE to the path of the file (or pass the path in as construction argument).
        3. Add your API key to the file plaintextly, without any spaces, quotes, etc.
"""
class ForecastClient:

    def __init__(self, sfn=SECRET_FILE):
        self.API_KEY: str
        try:
            self.API_KEY = open(sfn).read().strip()
        except FileNotFoundError:
            print("No secret file found. Please create one.")


    def get_forecast(self, future: date | int, current=date.today()) -> list[dict]:
        """
        Get the weather forecast for a given date range.

        NOTE:
            Calling this API may cost money when free credits are used.
                Cost: $0.0024 per difference in days  (future=5, then 5 * 0.0024 = $0.0120)

        :param future:   int — adds <future> days to current date, date — uses date directly
        :param current:  current date (default: today)
        :return:         returns the data from the API in JSON list by hours (see data/forcast_example.json)
        """
        # add days to current date
        if isinstance(future, int):
            future: date = current + timedelta(days=future)

        # parameters needed for the ML model
        params = {
            "include": "hours",
            "elements":
                "name,datetime,temp,feelslike,dew,humidity,precip,precipprob,"
                "preciptype,snow,snowdepth,windgust,windspeed,winddir,"
                "sealevelpressure,cloudcover,visibility,solarradiation,"
                "solarenergy,uvindex,sunazimuth,sunelevation,stations,source"
                }

        print(f"Getting forecast for {future} days from {current}...")

        # get response from API
        url = f"{BASE_URL}/{current}/{future}?key={self.API_KEY}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # convert the data to something more useful
        data = _parse_response(response.json())

        # TODO THIS IS FOR TESTING
        for d in data:
            print(d)
        with open("data/forecast_example.json", "w") as f:
            json.dump(data, f, indent=4)
        # TODO END TESTING

        return data


### TESTING ###
if __name__ == "__main__":
    test = ForecastClient()
    # test.get_forecast(1)
