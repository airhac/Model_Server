from PIL import Image
from django.http import HttpResponse
from django.shortcuts import render
from matplotlib import pyplot as plt
from rest_framework.parsers import JSONParser
from io import BytesIO

from modelapp.forms import ServerForm
from modelapp.image_model import Image_Model
from modelapp.models import Server_Model
from modelapp.video import Make_Video
# Create your views here.
from rest_framework.renderers import JSONRenderer
from rest_framework.views import APIView
from rest_framework.response import Response
import base64
import numpy as np
import json
class JSONResponse(HttpResponse):
    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] ='application/json'
        super(JSONResponse,self).__init__(content, **kwargs)

class UserView(APIView):
    model = Server_Model

    def post(self, request):
        # form받아옴
        data = request.data #data가 list으로 들어온다. 안에 dictionary 형태로 되어있음
        image_list = []

        for d in range(len(data)):
            temp1 = bytes(data[d]['image_base64'], 'ascii') #ascii 코드 형태로 byte를 변환해준다.
            temp = BytesIO(base64.b64decode(temp1))
            #BytesIO()는
            img = Image.open(temp)
            img_array = np.array(img)
            image_list.append(img_array)

        model = Image_Model()
        video = Make_Video()
        cuts = model.make_cut(image_list)
        image_len_list = video.image_len(cuts)
        # [[이미지,문자열길이],[이미지,문자열길이],...]

        # 리스트를 영상처리해주는 함수에 넣고 저장해줌
        # 반환값은 저장된 영상위치
        video_path = video.view_seconds(image_len_list)
        servermodel = Server_Model(image_num=data[0]['animate'], video=video_path)
        servermodel.save()

        return JSONResponse(data=video_path, status=200)  # 테스트용 Response

