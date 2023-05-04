import requests
import time

def get_qis(query):
    time.sleep(0.1)
    url = 'http://nar.staging.qis.search.yahoo.com:4080/v2/qis'
    params = {'intl': 'us', 'query': query}

    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Assuming the response is in JSON format
    else:
        print(f'Request failed with status code: {response.status_code}')
        return None
    

def extract_qis(qis):
    shopping_intent = 'shopping' in qis['intents']
    product_intent = 'product' in qis['intents']
    product = product_family = model = material = None
    if 'crf_product' in qis['tagging']:
        for annotation in qis['tagging']['crf_product']['annotations']:
            if annotation['tag'] == 'product':
                product = annotation['term']
            if annotation['tag'] == 'product_family':
                product_family = annotation['term']
            if annotation['tag'] == 'model':
                model = annotation['term']
            if annotation['tag'] == 'material':
                material = annotation['term']
    return {'product_intent': product_intent,
            'shopping_intent': shopping_intent,
            'product': product,
            'product_family': product_family,
            'model': model,
            'material': material}

            
