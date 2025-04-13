import requests


def fetch_and_save_file(url , path):
    r = requests.get(url);
    with open(path ,"w") as f:
        f.write(r.text);
        
    
    
url = "https://www.thenews.com.pk/latest/1243016-president-approves-justice-yahya-afridis-appointment-as-next-cjp";
fetch_and_save_file(url ,"data/news.html")
