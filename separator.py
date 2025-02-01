import pandas as pd

file = pd.read_csv("hand_landmarks_data.csv")

file_pivoted = file.pivot_table(index=['Frame'],
                                columns='Landmark_ID',
                                values=['Normalized_X', 'Normalized_Y'],
                                aggfunc='first')

file_pivoted.columns = [f'({col[0]}{col[1]})' for col in file_pivoted.columns]

file_pivoted = file_pivoted.reset_index()

print(file_pivoted.head())