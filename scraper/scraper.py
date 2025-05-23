from bs4 import BeautifulSoup
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import time
from multiprocessing import Process,cpu_count, Event
from headers import get_headers
import time 
from dotenv import load_dotenv
import os
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from logger_config import setup_logger
import signal


load_dotenv(verbose=True, override=True)
MONGODB_URI = os.getenv("MONGODB_URI")

# MongoDB client setup
logger = setup_logger('scraper')
try:
    client = MongoClient(MONGODB_URI)
    # Test the connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise


name = "scrap2023"
#do this for upto 2021 dec 30  
start_full_date = "05/01/2025"
end_full_date = "05/22/2025"

# db = client[name]
db = client["admin"]
collection = db["scraped_data"]
processors = cpu_count()

# Add a global shutdown event
shutdown_event = Event()

def get_collections(year):
    """Returns separate collections for floorsheet and date based on the instance year."""
    floorsheet_collection = db[f"floorsheet_{year}"]
    date_collection = db[f"date_{year}"]
    return floorsheet_collection, date_collection

async def update_date_collection(date_collection, date, current_page, failed_pages=[]):
    """Updates progress in the date collection asynchronously."""
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: date_collection.update_one(
                {"date": date},
                {"$set": {"current_page": current_page, "failed_pages": failed_pages}},
                upsert=True
            )
        )
        logger.info(f"Updated date collection for {date}: Page {current_page}, Failed pages: {failed_pages}")
    except Exception as e:
        logger.error(f"Error in update_date_collection: {e}")

async def fetch_page_async(session, date, page, url, cookies, headers, retries=3):
    """Fetches a single page asynchronously with retries."""
    delay = 1
    for attempt in range(retries):
        try:
            data = get_headers(date, page, data_only=True)
            timeout = ClientTimeout(total=30)  # 30 second timeout per request
            async with session.post(url, headers=headers, cookies=cookies, data=data, timeout=timeout) as response:
                if response.status == 200:
                    logger.info(f"Fetched page {page} for {date}")
                    return await response.text()
                elif response.status == 429:
                    logger.warning(f"Rate limited on page {page} for {date}. Waiting {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logger.error(f"Failed to fetch page {page} for {date}. HTTP {response.status}")
                    await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"Request exception on page {page} for {date}: {e}")
            if attempt < retries - 1:  # Don't sleep on last attempt
                await asyncio.sleep(delay)
            delay *= 2
    logger.error(f"Max retries reached for page {page} on {date}.")
    return None

def get_total_pages(html):
    """Extracts the total number of pages from the HTML response."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        records_span = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_PagerControl1_litRecords'})
        if records_span:
            text = records_span.get_text()
            start = text.find('Total pages: ') + len('Total pages: ')
            end = text.find(' ', start)
            total_pages = int(text[start:end])
            return total_pages
        return None
    except Exception as e:
        logger.error(f"Error in get_total_pages: {e}")
        return None

def parse_table_data(html):
    """Parses table data from the fetched page."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        table_data = []
        div_data = soup.find('div', id='ctl00_ContentPlaceHolder1_divData')
        if div_data:
            table = div_data.find('table')
            if table:
                tbody = table.find('tbody')
                if tbody:
                    for tr in tbody.find_all('tr'):
                        row_data = [td.text.strip() for td in tr.find_all('td')]
                        table_data.append(row_data)
        return table_data
    except Exception as e:
        print(f"Error in parse_table_data: {e}")
        return []

def transform_to_document(response_array, transaction_date):
    """Transforms parsed data into MongoDB documents."""
    try:
        return [
            {
                "_id": ObjectId(),
                "transaction": row[1],  # Using transaction as unique identifier
                "id": int(row[0]),
                "symbol": row[2],
                "buyer": row[3],
                "seller": row[4],
                "quantity": int(row[5].replace(",", "")),
                "rate": float(row[6].replace(",", "")),
                "amount": float(row[7].replace(",", "")),
                "transaction_date": datetime.strptime(transaction_date, "%m/%d/%Y").strftime("%Y-%m-%d"),
            }
            for row in response_array
        ]
    except Exception as e:
        logger.error(f"Error in transform_to_document: {e}")
        return []

async def insert_with_retry(collection, data, retries=3):
    """Inserts data into MongoDB with retries asynchronously, skipping existing transactions."""
    while retries > 0:
        try:
            # Filter out existing transactions
            transactions = [doc["transaction"] for doc in data]
            existing_transactions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(collection.find(
                    {"transaction": {"$in": transactions}},
                    {"transaction": 1}
                ))
            )
            
            existing_trans = set(doc["transaction"] for doc in existing_transactions)
            new_data = [doc for doc in data if doc["transaction"] not in existing_trans]
            
            if new_data:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.insert_many(new_data)
                )
                logger.info(f"Inserted {len(new_data)} new records, skipped {len(data) - len(new_data)} existing records")
            else:
                logger.info(f"Skipped {len(data)} existing records")
            return True
        except Exception as e:
            logger.error(f"Database insertion error: {e}")
            await asyncio.sleep(3)
            retries -= 1
    logger.error("Max retries reached for database insertion.")
    return False

async def save_progress(date_collection, date, current_page, failed_pages):
    """Saves the current progress to the date collection asynchronously."""
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: date_collection.update_one(
                {"date": date},
                {"$set": {"current_page": current_page, "failed_pages": failed_pages}},
                upsert=True
            )
        )
        print(f"Progress saved for {date}: Page {current_page}, Failed pages: {failed_pages}")
    except Exception as e:
        print(f"Error in save_progress: {e}")

def handle_interrupt(signum, frame):
    """Handles the KeyboardInterrupt to gracefully stop all processes."""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    shutdown_event.set()  # Set the event to signal all processes to stop

async def process_page(session, floorsheet_collection, date_collection, date, page, total_pages, url, cookies, headers):
    """Processes a single page of data."""
    try:
        html = await fetch_page_async(session, date, page, url, cookies, headers)
        if html:
            table_data = parse_table_data(html)
            if table_data:
                documents = transform_to_document(table_data, date)
                if documents:
                    success = await insert_with_retry(floorsheet_collection, documents)
                    if success:
                        await save_progress(date_collection, date, page, [])
                        return True
            else:
                logger.warning(f"No table data found on page {page} for {date}")
                return False
        logger.error(f"Failed to fetch page {page} for {date}")
        return False
    except Exception as e:
        logger.error(f"Error processing page {page} for {date}: {e}")
        return False

async def process_date_async(floorsheet_collection, date_collection, date, start_page, total_pages=None):
    """Processes a single date by scraping all pages concurrently."""
    url, cookies, headers, _ = get_headers(date, 1)
    failed_pages = []
    
    timeout = ClientTimeout(total=180)  # Increased timeout
    connector = aiohttp.TCPConnector(
        limit=10,
        limit_per_host=5,  # Limit per host to be safer
        enable_cleanup_closed=True,
        keepalive_timeout=30  # Shorter keepalive to free connections faster
    )
    
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            if total_pages is None:
                html = await fetch_page_async(session, date, 1, url, cookies, headers)
                if not html:
                    logger.error(f"Could not fetch first page for {date}")
                    return
                total_pages = get_total_pages(html)
                if not total_pages:
                    logger.error(f"Could not determine total pages for {date}")
                    return
                
                await update_date_collection(date_collection, date, start_page, failed_pages)

            # Adaptive batch sizing
            remaining_pages = total_pages - start_page + 1
            if remaining_pages <= 10:
                batch_size = 2  # Smaller batch size for few remaining pages
            elif remaining_pages <= 20:
                batch_size = 3
            else:
                batch_size = 5

            # Process pages in batches
            for i in range(start_page, total_pages + 1, batch_size):
                batch_end = min(i + batch_size, total_pages + 1)
                tasks = []
                
                for page in range(i, batch_end):
                    task = process_page(session, floorsheet_collection, date_collection, 
                                    date, page, total_pages, url, cookies, headers)
                    tasks.append(task)
                
                # Process batch with timeout
                try:
                    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60)
                    
                    # Track failed pages in this batch
                    for idx, success in enumerate(results):
                        if not success:
                            failed_page = i + idx
                            if failed_page not in failed_pages:
                                failed_pages.append(failed_page)
                                logger.warning(f"Failed to process page {failed_page} for {date}")
                    
                    # Update progress more frequently for remaining pages
                    if remaining_pages <= 20:
                        await update_date_collection(date_collection, date, batch_end - 1, failed_pages)
                        
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing batch starting at page {i} for {date}")
                    for page_num in range(i, batch_end):
                        if page_num not in failed_pages:
                            failed_pages.append(page_num)
                
                # Adaptive delay between batches
                delay = 1 if remaining_pages <= 10 else 2
                await asyncio.sleep(delay)
                
            # Retry failed pages individually
            if failed_pages:
                logger.info(f"Retrying {len(failed_pages)} failed pages for {date}")
                for failed_page in failed_pages[:]:  # Create a copy to modify during iteration
                    try:
                        success = await process_page(session, floorsheet_collection, date_collection,
                                                   date, failed_page, total_pages, url, cookies, headers)
                        if success:
                            failed_pages.remove(failed_page)
                            logger.info(f"Successfully retried page {failed_page} for {date}")
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error retrying page {failed_page} for {date}: {e}")
                
            # Final update
            await update_date_collection(date_collection, date, total_pages, failed_pages)
            
    except Exception as e:
        logger.error(f"Error processing date {date}: {e}")
        await update_date_collection(date_collection, date, start_page, failed_pages)
    finally:
        await connector.close()

async def start_process_async(start_date_obj, year):
    """Async version of start_process."""
    floorsheet_collection, date_collection = get_collections(year)
    floorsheet_collection = collection
    date_str = start_date_obj.strftime("%m/%d/%Y")
    progress = date_collection.find_one({"date": date_str}) or {}
    current_page = progress.get("current_page", 1)
    total_pages = progress.get("total_pages", None)
    logger.info(f"Processing date: {date_str}, starting at page {current_page}")
    await process_date_async(floorsheet_collection, date_collection, date_str, current_page, total_pages)

async def main_async(start_date, end_date, year, shutdown_event):
    """Async version of main function."""
    start_date_obj = datetime.strptime(start_date, "%m/%d/%Y")
    end_date_obj = datetime.strptime(end_date, "%m/%d/%Y")
    
    current_date = start_date_obj
    while current_date <= end_date_obj:
        if shutdown_event.is_set() or os.path.exists("KILL_SCRAPER"):
            logger.info("Shutdown requested or kill switch detected. Stopping processing...")
            break
        if shutdown_event.is_set():
            logger.info("Shutdown requested, stopping processing...")
            break
            
        try:
            await start_process_async(current_date, year)
        except Exception as e:
            logger.error(f"Error processing date {current_date}: {e}")
        current_date += timedelta(days=1)
        await asyncio.sleep(1)

def process_chunk(start_date, end_date, year):
    """Process a chunk of dates in a separate process."""
    try:
        # Set up signal handler in each process
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)
        
        asyncio.run(main_async(start_date, end_date, year, shutdown_event))
    except Exception as e:
        logger.error(f"Error in process_chunk: {e}")

def run_instance(start_date, end_date, year):
    """Runs an instance of the scraper using asyncio."""
    logger.info(f"Starting scraper for {year}: {start_date} to {end_date}")
    asyncio.run(main_async(start_date, end_date, year, shutdown_event))

def split_date_range(start_date, end_date, num_processes):
    """Split the date range into chunks for parallel processing, excluding weekends."""
    start = datetime.strptime(start_date, "%m/%d/%Y")
    end = datetime.strptime(end_date, "%m/%d/%Y")
    
    # Create list of valid dates (excluding weekends)
    valid_dates = []
    current = start
    while current <= end:
        if current.weekday() not in [4, 5]:  # Skip Friday (4) and Saturday (5)
            valid_dates.append(current)
        current += timedelta(days=1)
    
    # Calculate chunk size based on valid dates
    total_valid_days = len(valid_dates)
    chunk_size = total_valid_days // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    # Split valid dates into chunks
    chunks = []
    for i in range(0, total_valid_days, chunk_size):
        chunk_dates = valid_dates[i:i + chunk_size]
        if chunk_dates:
            chunks.append((
                chunk_dates[0].strftime("%m/%d/%Y"),
                chunk_dates[-1].strftime("%m/%d/%Y")
            ))
    
    # If we have fewer chunks than processors, adjust
    while len(chunks) < num_processes and len(chunks) > 1:
        # Split the largest chunk
        largest_chunk_idx = 0
        largest_chunk_size = 0
        for i, (start_str, end_str) in enumerate(chunks):
            start_date = datetime.strptime(start_str, "%m/%d/%Y")
            end_date = datetime.strptime(end_str, "%m/%d/%Y")
            size = len([d for d in valid_dates if start_date <= d <= end_date])
            if size > largest_chunk_size:
                largest_chunk_size = size
                largest_chunk_idx = i
        
        # Split the largest chunk into two
        if largest_chunk_size > 1:
            start_str, end_str = chunks[largest_chunk_idx]
            start_date = datetime.strptime(start_str, "%m/%d/%Y")
            end_date = datetime.strptime(end_str, "%m/%d/%Y")
            mid_date = start_date + timedelta(days=(end_date - start_date).days // 2)
            
            # Find the next valid date after mid_date
            while mid_date.weekday() in [4, 5]:
                mid_date += timedelta(days=1)
            
            chunks[largest_chunk_idx] = (start_str, mid_date.strftime("%m/%d/%Y"))
            chunks.append((
                (mid_date + timedelta(days=1)).strftime("%m/%d/%Y"),
                end_str
            ))
        else:
            break
    
    logger.info(f"Created {len(chunks)} chunks from {total_valid_days} valid trading days")
    return chunks

def setup_collections(year):
    """Sets up collections with proper indexes."""
    floorsheet_collection, date_collection = get_collections(year)
    
    # Create unique index on transaction field
    floorsheet_collection.create_index([("transaction", 1)], unique=True)
    
    # Create index on transaction_date for efficient queries
    floorsheet_collection.create_index([("transaction_date", 1)])
    
    return floorsheet_collection, date_collection

if __name__ == "__main__":
    # Set up signal handlers in main process
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    
    year = name
    logger.info(f"Number of processors: {processors}")
    start_time = time.time()
    
    # Setup collections with indexes
    floorsheet_collection, date_collection = setup_collections(year)
    
    instances = split_date_range(start_full_date, end_full_date, processors)
    
    processes = []
    try:
        for start, end in instances:
            if shutdown_event.is_set():
                break
                
            p = Process(target=process_chunk, args=(start, end, year))
            p.start()
            processes.append(p)
            
            if len(processes) >= processors:
                for p in processes:
                    p.join()
                processes = []
        
        # Wait for remaining processes
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in main process")
        shutdown_event.set()
        
        # Give processes a chance to shut down gracefully
        logger.info("Waiting for processes to finish...")
        for p in processes:
            p.join(timeout=10)  # Wait up to 10 seconds for each process
            if p.is_alive():
                logger.warning("Force terminating process...")
                p.terminate()
                p.join()
    
    finally:
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        logger.info("Scraping tasks completed or interrupted.")
        if os.path.exists("KILL_SCRAPER"):
            os.remove("KILL_SCRAPER") 