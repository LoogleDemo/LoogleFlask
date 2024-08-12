import google.generativeai as genai
import PIL.Image
import requests
import pandas as pd
import time
import requests

genai.configure(api_key=apikey)

df = pd.read_excel('loogle_test.xlsx')
keywords = []
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

for index, row in df.iterrows():
    productImgUrls = eval(row['productImgUrl']) 
    if isinstance(productImgUrls, list) and productImgUrls:
        productImgUrl = productImgUrls[0]
    else:
        raise ValueError("productImgUrl is not a valid list format.")

    print(productImgUrl)
    
    if productImgUrl:
        response = requests.get(productImgUrl, stream=True, verify=False)
        with open('temp.jpg', 'wb') as out_file:
            for chunk in response.iter_content(1024):
                out_file.write(chunk)

        userImageFile = PIL.Image.open('temp.jpg')
        response = model.generate_content([userImageFile, "너는 패션 분석가로서 100자 이내로 분석해주고 뉘앙스적인 것도 포함되었으면 해. 답변은 개조식의 긴 하나의 문장으로 부탁"])
        
        print(response.text)
        keywords.append(response.text)
        
        
        time.sleep(30)

df['keywordList'] = keywords
df.to_excel('loogle_updated.xlsx', index=False)
