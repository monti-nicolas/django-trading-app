import yfscreen as yfs
import pandas as pd
import argparse


def update_etf(save_path:str = './source/etf/', top_n:int = 500, region:str = 'us'):
    """Extracts ETF list to be considered for forecast and investment.

    Args:
        save_path (str): Path were .csv files with ETF details will be saved.
        top_n (int): Number of ETFs from the specified region that will be extracted after sorting by average traded
        volume descending. This is required to avoid extracting a full list of ETFs that will require long-running
        times to forecast.
        region (str): The market from which we want to extract ETFs data.

    Returns:
        None

    """

    # Create a query to filter for US ETFs
    filters = [
        ["eq", ["region", region]]
    ]

    # Create the query from filters
    query = yfs.create_query(filters)

    # Create payload for ETF security type
    # Set size to maximum (250) to get as many results as possible per page
    payload = yfs.create_payload(
        sec_type="etf",  # Specify ETF as security type
        query=query,
        size=250,  # Maximum results per request
        offset=0
    )

    # Get the first batch of results
    response = yfs.get_data(payload)

    # Convert to DataFrame
    all_etfs = pd.DataFrame(response)

    # Handle pagination to get all results
    target_count = 5000

    # Continue fetching if there are more results
    offset = 250
    while offset < target_count:
        payload['offset'] = offset
        response = yfs.get_data(payload)

        batch_df = pd.DataFrame(response)
        all_etfs = pd.concat([all_etfs, batch_df], ignore_index=True)

        offset += 250
        print(f"Fetched {len(all_etfs)} ETFs so far...")

    # Extract ticker symbols
    etf_tickers = all_etfs['symbol'].tolist()

    print(f"\nTotal ETFs retrieved: {len(etf_tickers)}")
    print(f"\nFirst 10 ETF tickers: {etf_tickers[:10]}")

    # Optionally save to CSV
    all_etfs.to_csv(save_path + 'us_etf_list.csv', index=False)
    print("\nFull ETF data saved to 'us_etf_list.csv'")

    # Sorting and selection process
    print("Available columns:")
    print(all_etfs.columns.tolist())
    print("\n")

    # Common volume column names in Yahoo Finance data:
    # 'averageDailyVolume3Month', 'averageDailyVolume10Day', 'volume'

    # Sort by average daily volume (3-month is most stable)
    if 'averageDailyVolume3Month.raw' in all_etfs.columns:
        volume_col = 'averageDailyVolume3Month.raw'
        print(f"[INFO] Using column name: {volume_col}")
    elif 'averageDailyVolume10Day' in all_etfs.columns:
        volume_col = 'averageDailyVolume10Day'
    elif 'volume' in all_etfs.columns:
        volume_col = 'volume'
    else:
        print("No volume column found. Available columns:")
        print(all_etfs.columns.tolist())
        volume_col = None

    if volume_col:
        # Remove ETFs with missing volume data
        etfs_with_volume = all_etfs[all_etfs[volume_col].notna()].copy()

        # Sort by volume descending
        etfs_sorted = etfs_with_volume.sort_values(by=volume_col, ascending=False)

        # Get top n
        top_n_etfs = etfs_sorted.head(top_n).copy()

        # Display some statistics
        print(f"Total ETFs with volume data: {len(etfs_with_volume)}")
        print(f"\nTop {top_n} ETFs selected based on {volume_col}")
        print(f"\nVolume range in top {top_n}:")
        print(f"  Highest: {top_n_etfs[volume_col].max():,.0f}")
        print(f"  Lowest: {top_n_etfs[volume_col].min():,.0f}")
        print(f"  Median: {top_n_etfs[volume_col].median():,.0f}")

        # Show top 10
        print(f"\nTop 10 ETFs by {volume_col}:")
        print(top_n_etfs[['symbol', 'longName', volume_col]].head(10).to_string(index=False))

        top_n_etfs.rename(columns={'symbol': 'ticker', 'longName': 'company_name'}, inplace=True)

        # Save to files
        top_n_etfs.to_csv(save_path + 'tickers-details.csv', index=False)
        print(f"\nTop {top_n} ETFs data saved to 'tickers-details.csv'")

        top_n_tickers = top_n_etfs['ticker']
        top_n_tickers.to_csv(save_path + 'tickers.csv', index=False)
        print(f"\nTop {top_n} ETFs data saved to 'tickers.csv'")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_n",
        required=True,
        type=int,
        help="Number of ETFs from the specified region that will be extracted."
    )

    args = vars(parser.parse_args())

    if args['top_n'] == '' or args['top_n'] is None:
        print("[INFO] Extracting ETF data with default number of tickers.")
        update_etf()
    else:
        print(f"[INFO] Extracting ETF data with {args['top_n']} number of tickers.")
        update_etf(
            top_n=args['top_n']
        )
