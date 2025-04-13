Here is a basic guide to using `requests` and `BeautifulSoup` for web scraping, along with some common syntaxes for different tasks:

### 1. **Installing the Required Libraries**
Before starting, install the necessary libraries if you haven't done so:

```bash
pip install requests beautifulsoup4
```

### 2. **Importing the Libraries**
```python
import requests
from bs4 import BeautifulSoup
```

### 3. **Making a GET Request with `requests`**
To fetch the content of a web page:

```python
url = 'https://example.com'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    html_content = response.text  # or response.content for raw bytes
else:
    print(f"Failed to retrieve content: {response.status_code}")
```

### 4. **Parsing the HTML with BeautifulSoup**
Once you have the page content, you can parse it using `BeautifulSoup`:

```python
soup = BeautifulSoup(html_content, 'html.parser')
```

### 5. **Navigating the DOM**

- **Find by Tag Name**

```python
# Find the first <h1> tag
h1_tag = soup.find('h1')

# Find all <p> tags
p_tags = soup.find_all('p')
```

- **Find by Class or ID**

```python
# Find by class name
element_with_class = soup.find(class_='classname')

# Find by ID
element_with_id = soup.find(id='element-id')
```

- **Find by Attributes**

```python
# Find element by any attribute
tag_with_attr = soup.find('div', {'data-example': 'value'})
```

### 6. **Extracting Text or Attributes**

- **Extracting Text from a Tag**

```python
text = h1_tag.get_text()  # or .text
```

- **Extracting Attribute Values**

```python
# Example: Extracting the 'href' from an <a> tag
link = soup.find('a')
href = link['href']
```

### 7. **Searching with CSS Selectors**
Use `select()` to find elements using CSS selectors:

```python
# Find all elements with the class 'example'
elements = soup.select('.example')

# Find elements using tag and class combinations
elements = soup.select('div.example')

# Find elements by ID
element = soup.select('#unique-id')

# Find nested elements
nested = soup.select('div.container p')
```

### 8. **Iterating Through Multiple Elements**

```python
# Get all <a> tags and print their href attributes
links = soup.find_all('a')
for link in links:
    print(link.get('href'))
```

### 9. **Handling Pagination or Multiple Pages**

You might need to scrape multiple pages by changing the URL dynamically:

```python
for page in range(1, 6):
    url = f'https://example.com/page/{page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Do your scraping here
```

### 10. **Handling Headers and User Agents**
Some websites block scraping requests unless you provide a user agent:

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

response = requests.get(url, headers=headers)
```

### 11. **Handling Cookies**

If the website requires cookies:

```python
cookies = {'session_id': 'example_session_id'}
response = requests.get(url, cookies=cookies)
```

### 12. **Posting Data**
If you need to send data (e.g., filling forms):

```python
data = {'username': 'user', 'password': 'pass'}
response = requests.post('https://example.com/login', data=data)
```

### 13. **Dealing with JavaScript-Rendered Content**
`requests` and `BeautifulSoup` cannot handle JavaScript-rendered content. For that, you may need to use tools like **Selenium** or **Playwright**.

For example, using **Selenium**:

```python
from selenium import webdriver

driver = webdriver.Chrome()
driver.get('https://example.com')

# Get the page source after JavaScript has run
html_content = driver.page_source

# Pass it to BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

driver.quit()
```

---

These are the main syntaxes you would use for web scraping with `requests` and `BeautifulSoup`. Each task might require more customization based on the website you are scraping, but this gives you a solid foundation.