import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "d23e6ab187f839bf309c8e6f53754027"

df = pd.read_csv('1999-present.csv')
unique_urls = df['url'].dropna().drop_duplicates().reset_index(drop=True)

print(f"📦 Starting scrape for {len(unique_urls)} unique game URLs...\n")

def clean_key(key):
    return key.strip().lower().replace(" ", "_").replace("*", "")

def extract_game_info(url):
    full_url = f"https://www.pro-football-reference.com{url}" if not url.startswith("http") else url
    wrapped_url = f"http://api.scraperapi.com?api_key={API_KEY}&url={full_url}"

    try:
        print(f"🌐 Scraping: {full_url}")
        r = requests.get(wrapped_url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        game_info = {'url': url}

        # Extract Game Info table
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if 'div_game_info' in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                table = comment_soup.find('table', {'id': 'game_info'})
                if table:
                    for row in table.find_all('tr'):
                        th = row.find('th')
                        td = row.find('td')
                        if th and td:
                            key = clean_key(th.text)
                            value = td.text.strip()
                            game_info[key] = value

        # Extract stadium
        stadium = None
        scorebox_meta = soup.find('div', class_='scorebox_meta')
        if scorebox_meta:
            for div in scorebox_meta.find_all('div'):
                strong_tag = div.find('strong')
                if strong_tag and 'Stadium' in strong_tag.text:
                    stadium_link = div.find('a')
                    if stadium_link:
                        stadium = stadium_link.text.strip()
        game_info['stadium'] = stadium
        print("📍 Extracted stadium:", stadium)

        return game_info

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return {'url': url, 'stadium': None}  # Ensure stadium key always present

def scrape_urls(urls, label="first pass"):
    print(f"\n🚀 Running {label} with {len(urls)} URLs...\n")
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(extract_game_info, url): url for url in urls}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if i % 50 == 0:
                    print(f"✅ {i+1}/{len(urls)} scraped")
            except Exception as e:
                print(f"❌ Thread error: {e}")
    return pd.DataFrame(results)

# 🔁 First pass
first_pass_df = scrape_urls(unique_urls)

# 🔁 Retry URLs missing stadium
failed_urls = first_pass_df[first_pass_df['stadium'].isna()]['url'].tolist()
retry_df = pd.DataFrame()
if failed_urls:
    print(f"\n🔁 Retrying {len(failed_urls)} failed URLs...\n")
    retry_df = scrape_urls(failed_urls, label="retry pass")

# 🧩 Combine scraped data
combined_scraped_df = pd.concat([first_pass_df.set_index('url'), retry_df.set_index('url')])
final_scraped_df = combined_scraped_df[~combined_scraped_df.index.duplicated(keep='last')].reset_index()

# 🧬 Merge into full dataset (both rows per game)
df = df.merge(final_scraped_df, on='url', how='left', suffixes=('', '_fixed'))

# Overwrite only missing values using fixed columns
for col in final_scraped_df.columns:
    if col != 'url':
        fixed_col = f"{col}_fixed"
        if fixed_col in df.columns:
            df[col] = df[col].combine_first(df[fixed_col])
            df.drop(columns=[fixed_col], inplace=True)

# 🔐 Final guarantee: Fill empty stadiums with 'MISSING'
df['stadium'] = df['stadium'].fillna('MISSING')

# 💾 Save
df.to_csv("1999_present_full_game_info.csv", index=False)
print("\n✅ Final file saved to 1999_present_full_game_info.csv")

# 🔍 Summary
missing_stadiums = df[df['stadium'] == 'MISSING']['url'].drop_duplicates()
print(f"\n🚫 Still missing stadium info for {len(missing_stadiums)} games:")
print(missing_stadiums.tolist())
