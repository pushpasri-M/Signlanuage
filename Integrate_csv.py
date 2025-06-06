import pandas as pd

# Load individual CSV files and add labels
wave = pd.read_csv('wave.csv')
wave['label'] = 'wave'

cool = pd.read_csv('cool.csv')
cool['label'] = 'cool'

me = pd.read_csv('Me.csv')
me['label'] = 'Me'

peace = pd.read_csv('Peace.csv')
peace['label'] = 'Peace'

stop = pd.read_csv('Stop.csv')
stop['label'] = 'Stop'

# Combine all the dataframes into one
combined_data = pd.concat([wave, cool, me, peace, stop])

# Save the combined dataframe to a new CSV
combined_data.to_csv('combined_data.csv', index=False)

print("Data combined successfully!")
