import requests

"""
Notes, in bern 2 env run: python3 server.py. 
This runs it without neural normalizer on port 5000, which works. 
Will still recognize standard drug names, common diseases and species mutations, 
but will not link all of these and may struggle with acronyms etc. 
...With the neural normalizer crashes it. 
"""

def query_plain(text, url="http://localhost:5000/plain"):
    return requests.post(url, json={'text': text}).json()

if __name__ == '__main__':
    text = "Tazobactum"
    print(query_plain(text))


