import torch
import numpy as np
from collections import Counter
from PIL import Image
from django.conf import settings
import os
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=30, autoshape = False)
model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR,'5s.pt'), map_location = 'cpu')['model'].state_dict())
model = model.autoshape()
# model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model = 'best.pt', map_location = 'cpu')
model = model.to(device).eval()
classes = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper spot leaf', 'Bell_pepper leaf',
          'Blueberry leaf', 'Cherry leaf', 'Corn Gray spot leaf', 'Corn blight leaf', 'Corn rust leaf',
          'Peach leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Potato leaf', 'Raspberry leaf', 
          'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf',
          'Tomato Septoria spot leaf', 'Tomato bacterial spot leaf', 'Tomato late blight leaf', 'Tomato mosaic virus leaf', 
          'Tomato yellow virus leaf', 'Tomato leaf', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 
          'grape black rot leaf', 'grape leaf']

def predict(im):
    # file = request.files['file']
    # img_bytes = file.read()
    # start=time.time()
    # im = Image.open(io.BytesIO(img_bytes))
    # print("request time   ",time.time()-start)
    h, w = im.size

    im = im.resize((32,32))
    # start=time.time()
    print(im)
    result=model(np.array(im))
    # print("inference time ",time.time()-start)
    result=result.xyxy[0].cpu().numpy()
    result=result.tolist()
    # list containing dictionary of all bounding box of individual image
    out={'boxes' : []}    
    all_classes = []
    for i in range(len(result)):
        temp={}  
        temp['box']= result[i][:4]
        modify= [w/416, h/416, w/416, h/416]
        temp['box'] = [ int(temp['box'][j] * modify[j]) for j in range(4) ]
        
        temp['confidence']=float(result[i][4])
        temp['class']=classes[ int(result[i][5])]
        all_classes.append(temp['class'])	
        out['boxes'].append(temp)
    
    freq = Counter(all_classes).most_common(1)[0][0]
    print(freq)
    out['plant_name'] = freq.split()[0]
    if len(freq.split()) == 2:
        out['disease'] = False
        out['disease_name'] = ''
    else:
        out['disease'] = True
        out['disease_name'] = ' '.join(freq.split()[1:-1])

    return out

# if __name__ == "__main__":
#     file = request.files['file']
#     img_bytes = file.read()
#     im = Image.open(io.BytesIO(img_bytes))
#     predict(im)