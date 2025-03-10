import os
import time
import csv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Set the output directory
output_dir = os.path.join("..", "data")

# Create output directories
image_folder = os.path.join(output_dir, "chair_images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Set up Chrome options
options = Options()
options.add_argument("--disable-notifications")
options.add_argument("--start-maximized")

# Set up the driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Create CSV file
csv_file = os.path.join(output_dir, 'pinterest_chairs.csv')
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Pinterest ID', 'Name', 'Description', 'Alt Text', 'URL'])

def scrape_pinterest(num_images=500):
    # Go directly to the search page for designer chairs
    print("Navigating to Pinterest search page...")
    driver.get("https://www.pinterest.com/search/pins/?q=vitra%20chair%20design")
    time.sleep(5)  # Wait for page to load
    
    # Scroll to load more pins
    pin_data = []
    scroll_count = 0
    max_scrolls = 100  # Increased for 500 images
    
    while len(pin_data) < num_images and scroll_count < max_scrolls:
        # Find all pins with alt text
        pin_elements = driver.find_elements(By.CSS_SELECTOR, "div[data-test-id='pin-with-alt-text']")
        
        # Process newly found pins
        current_count = len(pin_data)
        for pin in pin_elements:
            if len(pin_data) >= num_images:
                break
                
            try:
                # Find the <a> element and get the pin ID
                a_tag = pin.find_element(By.TAG_NAME, "a")
                href = a_tag.get_attribute("href")
                
                # Extract pin ID from href
                if "/pin/" in href:
                    pin_id = href.split("/pin/")[1].split("/")[0]
                    
                    # Check if we already have this pin
                    if not any(p[0] == pin_id for p in pin_data):
                        # Get alt text
                        alt_text = a_tag.get_attribute("aria-label")
                        
                        pin_data.append((pin_id, alt_text))
                        print(f"Found pin: {pin_id} - {alt_text}")
            except Exception as e:
                print(f"Error extracting pin data: {e}")
        
        # If we found new pins, report progress
        if len(pin_data) > current_count:
            print(f"Found {len(pin_data)}/{num_images} pins")
            
        # If we haven't found enough pins, scroll down and try again
        if len(pin_data) < num_images:
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)
            scroll_count += 1
            print(f"Scrolling (attempt {scroll_count}/{max_scrolls})...")
    
    print(f"Extracted data for {len(pin_data)} pins")
    
    # Visit each pin page and get the image URL and additional data
    for idx, (pin_id, alt_text) in enumerate(pin_data):
        try:
            # Visit the pin page
            pin_url = f"https://www.pinterest.com/pin/{pin_id}/"
            print(f"Visiting pin page {idx+1}/{len(pin_data)}: {pin_url}")
            driver.get(pin_url)
            
            # Wait for the image to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-test-id='pin-closeup-image'] img"))
                )
            except TimeoutException:
                print("Timeout waiting for image to load")
            
            # Get the chair name
            name = ""
            try:
                name_element = driver.find_element(By.CSS_SELECTOR, "div[data-test-id='pinTitle'] h1")
                name = name_element.text.strip()
            except NoSuchElementException:
                print("Name element not found")
            
            # Get the description
            description = ""
            # First try the main pin description
            try:
                desc_element = driver.find_element(By.CSS_SELECTOR, "div[data-test-id='main-pin-description-text'] span")
                description = desc_element.text.strip()
            except NoSuchElementException:
                # Try the alternative location
                try:
                    accordion = driver.find_element(By.CSS_SELECTOR, "div[data-test-id='CloseupDetails'] div[data-test-id='accordion-panel']")
                    description = accordion.text.strip()
                except NoSuchElementException:
                    print("Description element not found")
            
            # Find the image element
            img_url = ""
            try:
                img_element = driver.find_element(By.CSS_SELECTOR, "div[data-test-id='pin-closeup-image'] img")
                img_url = img_element.get_attribute("src")
            except NoSuchElementException:
                print("Image element not found")
            
            # Download the image
            if img_url:
                # Create a filename using the pin ID
                img_filename = f"{pin_id}.jpg"
                img_path = os.path.join(image_folder, img_filename)
                
                # Download the image
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://www.pinterest.com/'
                }
                
                response = requests.get(img_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded image: {img_filename}")
                    
                    # Write data to CSV
                    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([pin_id, name, description, alt_text, img_url])
                    print(f"Added data to CSV for pin {pin_id}")
                        
                else:
                    print(f"Failed to download image: {response.status_code}")
            else:
                print("No image URL found")
                
        except Exception as e:
            print(f"Error processing pin {pin_id}: {e}")
            
        # Add a delay to avoid being blocked
        time.sleep(2)

try:
    # Run the scraper
    scrape_pinterest(200)  # Change this number to scrape more images
    print("\nScraping complete!")
    print(f"Image data saved to: {csv_file}")
    print(f"Images downloaded to: {image_folder}")
    
except Exception as e:
    print(f"Error: {e}")
    
finally:
    # Close the browser
    driver.quit()
    print("Browser closed.")