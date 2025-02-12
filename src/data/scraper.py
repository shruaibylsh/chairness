from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def setup_driver():
    # Use ChromeDriverManager to automatically manage the driver version
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scroll_page(driver, scroll_pause=3, scroll_times=3):
    """Scrolls down the page to load dynamic content."""
    for i in range(scroll_times):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

def extract_data(driver):
    """Extract images and basic metadata from the page."""
    images = driver.find_elements(By.TAG_NAME, "img")
    
    data = []
    for img in images:
        src = img.get_attribute("src")
        alt = img.get_attribute("alt")
        
        # TODO: add more refined logic to extract style, material, etc.
        metadata = {
            "image_url": src,
            "description": alt
        }
        data.append(metadata)
    return data

def main():
    driver = setup_driver()
    
    # Open Pinterest's search page for "chair"
    pinterest_url = "https://www.pinterest.com/search/pins/?q=chair"
    driver.get(pinterest_url)
    
    # Wait for initial content to load
    time.sleep(5)
    
    # Scroll to load more results
    scroll_page(driver, scroll_pause=3, scroll_times=3)
    
    # Extract image data and metadata
    scraped_data = extract_data(driver)
    
    # Example testing: Print out the data
    for idx, item in enumerate(scraped_data):
        print(f"Item {idx+1}:")
        print(" Image URL:", item["image_url"])
        print(" Description:", item["description"])
        print("-" * 40)
    
    # Close the browser
    driver.quit()

if __name__ == "__main__":
    main()
