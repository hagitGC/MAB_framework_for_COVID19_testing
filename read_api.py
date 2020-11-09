import pandas as pd
import urllib.request
import json
headers = ['_id', 'test_date', 'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'corona_result', 'age_60_and_above', 'gender', 'test_indication']
url = 'https://data.gov.il/api/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&offset=0'

fileobj = urllib.request.urlopen(url)
raw_data = fileobj.read().decode('utf-8')
url_data_dict = json.loads(raw_data)
print(json.dumps(url_data_dict, indent = 4, sort_keys=True))
data = url_data_dict['result']['records']
data_df = pd.DataFrame(data)
for i in range(1, 1500):
    offset = i*100
    next = 'https://data.gov.il/api/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&offset={}'.format(offset)
    print (next)
    # req = urllib.request.Request('http://www.pretend_server.org')
    try: page = urllib.request.urlopen(next).read().decode('utf-8')
	except urllib.error.URLError as e:
		print(e.reason)
		#break
    # raw_data = req.read().decode('utf-8')
    url_data_dict = json.loads(page) #(raw_data)
    next_data = url_data_dict['result']['records']
    next_data_df = pd.DataFrame(next_data)
    data_df = pd.concat([data_df, next_data_df])

print (data_df.shape)
filename =  'C:/Users/hagit/pycharmprojects/covid_gov_data/data_df_{}'.format(offset)
data_df.to_pickle(filename)