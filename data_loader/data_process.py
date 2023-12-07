from __future__ import unicode_literals, print_function, division
import pandas as pd
import os


def normalizeString(df, lang):
    sentence = df[lang].str.lower()
    if lang =='Problems':
        sentence = sentence.str.replace('[^A-Za-z\s]+', ' ',regex=True)
    else:
        sentence = sentence.str.replace('[^A-Za-z\s]+', '',regex=True)
        sentence = sentence.str.replace('x{3,}', '', regex=True)

    sentence = sentence.str.strip()
    sentence = sentence.str.normalize('NFD')
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence

def stopword(df,lang):
    sentence = df[lang]

csv_dir = 'D:/data/iuct/origin/'
reports = pd.read_csv(os.path.join(csv_dir,'indiana_reports.csv'))
projection = pd.read_csv(os.path.join(csv_dir,'indiana_projections.csv'))

print(reports['impression'].isna().sum())
reports = reports.fillna('null')

keyword = normalizeString(reports,'Problems')
caption = normalizeString(reports,'impression')



ndata = pd.DataFrame({'uid':[],'image_name_0':[],'image_name_1':[],'keyword':[],'caption':[]})

ndata['uid'] = reports['uid']
ndata['keyword'] = keyword
ndata['caption'] = caption

image1 = []
image2 = []
for i in range(0,len(ndata)):
    uid =ndata.loc[i,'uid']
    query_str = "uid==" + str(uid)
    search_data = projection.query(query_str)['filename'].values.tolist()
    if len(search_data) == 2 :
        image1.append(search_data[0])
        image2.append(search_data[1])
    else:
        image1.append(search_data[0])
        image2.append('null')

ndata['image_name_0'] = image1
ndata['image_name_1'] = image2


ndata.to_csv(os.path.join(csv_dir,'new_data.csv'),index=False)

