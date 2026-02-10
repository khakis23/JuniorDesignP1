import os


SECRET_FILE = "secrets"

class ForecastClient:

    def __init__(self, sfn=SECRET_FILE):
        self.API_KEY: str
        try:
            self.API_KEY = open(sfn).read()
        except FileNotFoundError:
            print("No secret file found. Please create one.")




if __name__ == "__main__":
    test = ForecastClient()
