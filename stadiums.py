import pandas as pd

# Load your CSV file
df = pd.read_csv('Stadium_Fields_Filled_from_URL.csv')

# Filter rows with missing latitude or longitude
missing_coords_df = df[df[['latitude', 'longitude']].isna().any(axis=1)]

# Save the URLs of those rows into a new CSV
missing_coords_df[['url']].to_csv('missing_coordinates_urls.csv', index=False)

# Print summary
print(f"Number of rows with missing coordinates: {len(missing_coords_df)}")
print("URLs saved to 'missing_coordinates_urls.csv'")
