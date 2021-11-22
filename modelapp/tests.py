import base64
import numpy as np
import json

from PIL import Image


def base64_encode(image_list):
    base64_list = []
    for i in range(len(image_list)):
        t = base64.b64encode(image_list[i])
        base64_list.append(t)
    return base64_list

im = Image.open('C:\\Users\\oooh3\\PycharmProjects\\Model_Server\\media\\ani_image\\00279.jpg')
img_array = np.array(im)
base64_list = base64_encode(img_array)
#이미지 리스트를 cut분리 해주는 함수에 넣어줌
#반환값은 cut이미지 - cut 마다 array로 반환 된다.
json_dict = {}
for i in range(len(base64_list)):
    json_dict[i+1] = base64_list[i].decode('ascii')

print(type(json_dict))
json_data = json.dumps(json_dict)

json_data = json.loads(json_data)
array_list = []
dict_list = list(json_data.values())
for i in range(len(dict_list)):
    t = bytes(dict_list[i], encoding='ascii')
    r = base64.decodebytes(t)
    q = np.frombuffer(r, dtype=np.int8)
    array_list.append(q)

print(array_list)
