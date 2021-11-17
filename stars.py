import bs4, requests
from listaRepo import repo

for r in repo:
    try:
        response = requests.get(r)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        trovato = soup.find('div.js-toggler-container')
        ecco = soup.find("a", attrs={"class": "social-count js-social-count"})
        print(ecco.get_text())
    except:
        print("####")