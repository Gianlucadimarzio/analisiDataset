from google_play_scraper import app
import csv
from packages import package

for p in package:
    try:
        result = app (
            app_id= p
        )
        print(result["url"], " ",  result["genre"], " ", result["ratings"])
    except:
        print("####")