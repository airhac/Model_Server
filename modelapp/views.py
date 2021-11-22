from PIL import Image
from django.http import HttpResponse
from io import BytesIO
from modelapp.new_model import New_Model
from modelapp.models import Server_Model
from modelapp.speech_bubble_model import ComicFrameBook
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
    #model = Server_Model() 본 모델
    new_model = New_Model()
    make_video = Make_Video()
    def post(self, request):
        # form받아옴
        data = request.data #data가 list으로 들어온다. 안에 dictionary 형태로 되어있음
        image_list = []
        t_c = data[0]['toon_comic']
        l_r = data[0]['left_right']
        ani_effect = data[0]['ani_effect']
        tran_effect = data[0]['transition_effect']

        for d in range(len(data)):
            temp1 = bytes(data[d]['image_base64'], 'ascii') #ascii 코드 형태로 byte를 변환해준다.
            temp = BytesIO(base64.b64decode(temp1))
            #BytesIO()는
            img = Image.open(temp)
            img_array = np.array(img)
            image_list.append(img_array)

            # toon 방식
        if t_c == 'T':
            # 전처리
            labels_cut, labels_bubble = self.new_model.image_preproc(image_list)
            # 컷 분리
            bubbles, centroids, bubble_centers = self.new_model.make_cut_bubble(image_list, labels_bubble, l_r, is_bubble=True)
            cuts, centroids_cut, polygons = self.new_model.make_cut_bubble(image_list, labels_cut, l_r, is_bubble=False)
            # 객체 생성과 동시에 말풍선과 컷을 매칭합니다.
            framebook = ComicFrameBook(ani_effect, bubbles, cuts, polygons, bubble_centers, page_len=len(image_list))
            img_list = framebook.makeframe_proc()

        # [[이미지,문자열길이],[이미지,문자열길이],...]

        # 리스트를 영상처리해주는 함수에 넣고 저장해줌
        # 반환값은 저장된 영상위치
        video_path = self.make_video.new_view_seconds(img_list ,t_c, ani_effect, tran_effect)
        servermodel = Server_Model(image_num=data[0]['animate'], video=video_path)
        servermodel.save()

        return JSONResponse(data=video_path, status=200)  # 테스트용 Response


