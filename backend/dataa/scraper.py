import requests
from bs4 import BeautifulSoup

def fetch_job_data(query="Data Scientist", location=""):
    url = f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={query}&locT=C&locId=&locKeyword={location}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    jobs = []
    for job_card in soup.find_all("div", class_="jobContainer"):
        title = job_card.find("a", class_="jobLink").get_text(strip=True)
        company = job_card.find("div", class_="jobEmpolyerName").get_text(strip=True) if job_card.find("div", class_="jobEmpolyerName") else "Unknown"
        location = job_card.find("span", class_="loc").get_text(strip=True) if job_card.find("span", class_="loc") else "Unknown"
        
        jobs.append({
            "title": title,
            "company": company,
            "location": location
        })

    return jobs
