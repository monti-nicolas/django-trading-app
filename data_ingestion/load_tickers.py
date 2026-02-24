import yfscreen as yfs
import pandas as pd
import argparse
from pathlib import Path


def update_tickers(save_dir:str = './source/', top_n:int = 500, region:str = 'us', sec_type:str = 'etf'):
    """Extracts security list (ETF or stocks) from the specified region to be considered for forecast and investment.

    Args:
        save_dir (str): Main directory were .csv files with ticker details will be saved. Subdirectories will be
        created based on region and security type selected.
        top_n (int): Number of tickers that will be extracted after sorting by average traded volume descending.
        This is required to avoid extracting a full list of tickers that will require long-running
        times to forecast.
        region (str): The market from which we want to extract tickers data.
        sec_type (str): Type of securities we want to extract, either 'etf' or 'equity'.

    Returns:
        None

    """

    if sec_type == 'equity':
        sec_name = 'stock'
    else:
        sec_name = 'etf'

    save_path = save_dir + region + '/' + sec_name + '/'

    # Create directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Create a query to filter for region
    filters = [
        ["eq", ["region", region]]
    ]

    # Create the query from filters
    query = yfs.create_query(filters)

    # Set size to maximum (250) to get as many results as possible per page
    payload = yfs.create_payload(
        sec_type=sec_type,
        query=query,
        size=250,  # Maximum results per request
        offset=0
    )

    # Get the first batch of results
    response = yfs.get_data(payload)

    # Convert to DataFrame
    all_tickers = pd.DataFrame(response)

    # Handle pagination to get all results
    target_count = 8000

    # Continue fetching if there are more results
    offset = 250
    while offset < target_count:
        payload['offset'] = offset
        response = yfs.get_data(payload)

        batch_df = pd.DataFrame(response)

        # Check if we got any new data
        if len(batch_df) == 0:
            print(f"[INFO] No more records left to retrieve.")
            break

        all_tickers = pd.concat([all_tickers, batch_df], ignore_index=True)

        offset += 250
        print(f"Fetched {len(all_tickers)} tickers so far...")

    # Extract ticker symbols
    tickers_symbols = all_tickers['symbol'].tolist()

    print(f"\nTotal tickers retrieved: {len(tickers_symbols)}")
    print(f"\nFirst 10 tickers: {tickers_symbols[:10]}")

    # Optionally save to CSV
    all_tickers.to_csv(save_path + 'raw_tickers_extract.csv', index=False)
    print("\n[INFO] Full tickers data saved to 'raw_tickers_extract.csv'")

    # Sorting and selection process
    # print("[INFO] Available columns:")
    # print(all_tickers.columns.tolist())
    # print("\n")

    # Common volume column names in Yahoo Finance data:
    # 'averageDailyVolume3Month', 'averageDailyVolume10Day', 'volume'

    # Sort by average daily volume (3-month is most stable)
    if 'averageDailyVolume3Month.raw' in all_tickers.columns:
        volume_col = 'averageDailyVolume3Month.raw'
        print(f"[INFO] Using column name: {volume_col}")
    elif 'averageDailyVolume10Day' in all_tickers.columns:
        volume_col = 'averageDailyVolume10Day'
    elif 'volume' in all_tickers.columns:
        volume_col = 'volume'
    else:
        print("[INFO] No volume column found. Available columns:")
        print(all_tickers.columns.tolist())
        volume_col = None

    if volume_col:
        # Remove tickers with missing volume data
        tickers_with_volume = all_tickers[all_tickers[volume_col].notna()].copy()

        # Sort by volume descending
        tickers_sorted = tickers_with_volume.sort_values(by=volume_col, ascending=False)

        # Get top n
        top_n_tickers = tickers_sorted.head(top_n).copy()

        # Display some statistics
        print(f"Total tickers with volume data: {len(tickers_with_volume)}")
        print(f"\nTop {top_n} tickers selected based on {volume_col}")
        print(f"\nVolume range in top {top_n}:")
        print(f"  Highest: {top_n_tickers[volume_col].max():,.0f}")
        print(f"  Lowest: {top_n_tickers[volume_col].min():,.0f}")
        print(f"  Median: {top_n_tickers[volume_col].median():,.0f}")

        # Show top 10
        print(f"\nTop 10 tickers by {volume_col}:")
        print(top_n_tickers[['symbol', 'longName', volume_col]].head(10).to_string(index=False))

        top_n_tickers.rename(columns={'symbol': 'ticker', 'longName': 'company_name'}, inplace=True)

        # Save to files
        top_n_tickers.to_csv(save_path + 'tickers-details.csv', index=False)
        print(f"\nTop {top_n} tickers data saved to 'tickers-details.csv'")

        top_n_tickers = top_n_tickers['ticker']
        top_n_tickers.to_csv(save_path + 'tickers.csv', index=False)
        print(f"\nTop {top_n} tickers data saved to 'tickers.csv'")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_n",
        required=True,
        type=int,
        help="Number of tickers from the specified region that will be extracted."
    )

    # parser.add_argument(
    #     "--region",
    #     required=True,
    #     type=str,
    #     help="Region from where to extract tickers data."
    # )
    #
    # parser.add_argument(
    #     "--sec_type",
    #     required=True,
    #     type=str,
    #     help="Security Type: Either 'etf' or 'equity'."
    # )

    args = vars(parser.parse_args())

    regions = ['us', 'au']
    sec_types = ['etf', 'equity']

    for a_region in regions:
        for a_sec_type in sec_types:
            update_tickers(
                top_n=args['top_n'],
                region=a_region,
                sec_type=a_sec_type
            )

    # Combine all data sources
    print("[INFO] Combining all data sources...")
    filenames = ['tickers-details.csv', 'tickers.csv']

    for a_filename in filenames:
        combine_df = pd.DataFrame()
        for a_region in regions:
            for a_sec_type in sec_types:
                if a_sec_type == 'equity':
                    a_sec_name = 'stock'
                else:
                    a_sec_name = 'etf'

                a_path = './source/' + a_region + '/' + a_sec_name + '/' + a_filename
                df = pd.read_csv(a_path)
                combine_df = pd.concat([combine_df, df], ignore_index=True)

        # Save to files

        # Create directory if it doesn't exist
        combined_path = './source/combined/combined/'
        Path(combined_path).mkdir(parents=True, exist_ok=True)

        combine_df.to_csv(combined_path + a_filename, index=False)
        print(f"Tickers data saved to {a_filename}")
        print(f"\nTotal tickers retrieved: {len(combine_df)}")
