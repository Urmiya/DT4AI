import requests
import json

# API Key (Replace 'your_api_key' with your actual Financial Modeling Prep API key)
api_key = '72db1ad9bf86aa308c27cc9d148ca17c'

# Function to fetch ETFs and Funds
def fetch_etfs_and_funds(api_key):
    etf_url = f"https://financialmodelingprep.com/api/v3/etf/list?apikey={api_key}"
    fund_url = f"https://financialmodelingprep.com/api/v3/mutual_fund/list?apikey={api_key}"

    # Fetch ETF data
    etf_response = requests.get(etf_url)
    etfs = etf_response.json() if etf_response.status_code == 200 else []

    # Fetch Mutual Fund data
    fund_response = requests.get(fund_url)
    funds = fund_response.json() if fund_response.status_code == 200 else []

    # Filtering for relevance might involve checking descriptions or specific metadata
    # For now, we will print a subset of each due to the large amount of data typically returned
    print("ETFs:")
    for etf in etfs[:20]:  # print first 20 ETFs as an example
        print(f"{etf['symbol']} - {etf['name']}")

    print("\nMutual Funds:")
    for fund in funds[:20]:  # print first 20 Funds as an example
        print(f"{fund['symbol']} - {fund['name']}")

# Run the function
fetch_etfs_and_funds(api_key)
