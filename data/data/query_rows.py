import os
import pandas as pd

"""
use this file to get the number of rows in each csv file in the data directory

helpful for figuring out which files to use in preprocessing.

last run: nov 24:
celebrity_and_pop_culture.csv: 235 rows
food_and_dining.csv: 14 rows
science_and_technology.csv: 86 rows
learning_and_educational.csv: 35 rows
business_and_entrepreneurs.csv: 114 rows
arts_and_culture.csv: 111 rows
other_hobbies.csv: 23 rows
sports_and_gaming.csv: 584 rows
youth_and_student_life.csv: 1 rows
music.csv: 152 rows
pop_culture.csv: 711 rows
family.csv: 23 rows
gaming.csv: 45 rows
sports.csv: 305 rows
fitness_and_health.csv: 66 rows
daily_life.csv: 236 rows
travel_and_adventure.csv: 5 rows
diaries_and_daily_life.csv: 150 rows
film_tv_and_video.csv: 165 rows
fashion_and_style.csv: 39 rows
relationships.csv: 11 rows
news_and_social_concern.csv: 315 rows

"""

def get_csv_row_counts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(current_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            print(f"{csv_file}: {row_count:,} rows")
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")

get_csv_row_counts()