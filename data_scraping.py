from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from bs4 import BeautifulSoup
import os

for_loop_counter = 1

csv_path = "./unscraped_data.csv"
csv_output_path = "./data.csv"

df = pd.read_csv(csv_path)

new_columns = [
    "TIME_ORIGIN", "TEMPERATURE_ORIGIN", "DEW_POINT_ORIGIN", "HUMIDITY_ORIGIN", "WIND_ORIGIN",
    "WIND_SPEED_ORIGIN", "WIND_GUST_ORIGIN", "PRESSURE_ORIGIN", "PRECIP_ORIGIN", "CONDITION_ORIGIN",
    "TIME_DEST", "TEMPERATURE_DEST", "DEW_POINT_DEST", "HUMIDITY_DEST", "WIND_DEST",
    "WIND_SPEED_DEST", "WIND_GUST_DEST", "PRESSURE_DEST", "PRECIP_DEST", "CONDITION_DEST"
]

for col in new_columns:
    df[col] = ""

output_dir = os.path.dirname(csv_output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(csv_output_path):
    df.to_csv(csv_output_path, index=False)

driver = webdriver.Chrome()

def fetch_weather_data(city_name, city_code, date):
    url = f'https://www.wunderground.com/history/daily/us/{city_code}/{city_name}/date/{date}'
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tables = soup.find_all("table")
        for table in tables:
            new_table = pd.read_html(str(table))[0]
            if len(new_table.columns) >= len(new_columns) // 2:
                weather_data = new_table.iloc[11]
                return weather_data
    except Exception as e:
        print(f"Error fetching weather data for {city_name}, {city_code} on {date}: {e}")
        return None

for index, row in df.iterrows():
    retry_count = 0
    max_retries = 2
    while retry_count < max_retries:
        try:
            flight_date = row["FL_DATE"]
            origin_city_row = row["ORIGIN_CITY"]
            dest_city_row = row["DEST_CITY"]

            origin_city = origin_city_row.lower().split(", ")
            dest_city = dest_city_row.lower().split(", ")

            origin_city[0] = origin_city[0].split("/")[0]
            dest_city[0] = dest_city[0].split("/")[0]

            origin_city_name = origin_city[0].replace(" ", "-")
            origin_city_code = origin_city[1]
            weather_data_origin = fetch_weather_data(origin_city_name, origin_city_code, flight_date)

            if weather_data_origin is not None:
                for i, col in enumerate(new_columns[:10]):
                    df.at[index, col] = weather_data_origin[i]

            dest_city_name = dest_city[0].replace(" ", "-")
            dest_city_code = dest_city[1]
            weather_data_dest = fetch_weather_data(dest_city_name, dest_city_code, flight_date)
            print(for_loop_counter)
            for_loop_counter += 1

            if weather_data_dest is not None:
                for i, col in enumerate(new_columns[10:]):
                    df.at[index, col] = weather_data_dest[i]

            df.to_csv(csv_output_path, index=False)
            break
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}. Retry {retry_count}/{max_retries}.")
            if retry_count == max_retries:
                print("Max retries reached. Moving to next row.")
                break

driver.quit()
