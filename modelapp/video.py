import cv2.cv2 as cv2
from easyocr import easyocr
import datetime

class Make_Video():
    # 이미지 길이처리
    def image_len(self,cut_list):
        cut_img_list = []
        # 이미지를 자른다.
        for cut in cut_list:
            txt_len = self.img_text_easyocr(cut)
            # 리스트에 순서대로 잘라서 cut image, 글자수 순으로 추가
            cut_img_list.append([cut, txt_len])
        # cut의 [이미지,글자수]의 리스트 반환
        return cut_img_list

    # 인식률이 좋은 easyocr버전 이미지 받아 글자수 반환해주는 함수
    def img_text_easyocr(self,img):
        # image = Image.fromarray(np.uint8(cm.plasma(img) * 255))
        # img_resize = image.resize((int(image.width / 2), int(image.height / 2)))
        # 인식 언어 설정
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
        # 이미지를 받아 문자열 리스트를 반환해줌
        result = reader.readtext(img, detail=0)
        # 리스트 원소 합쳐서 문자여 총 길이 확인
        text_result = " ".join(result)
        text_result_len = len(text_result)
        print("길이:" + str(len(text_result)))
        print(text_result)
        # 문자열 길이 반환
        return text_result_len

    # [이미지,글자수]의 리스트를 받아 영상으로 만들고 저장하는 함수
    def view_seconds(self,image_list):
        # 영상이름 오늘 날자와 시간으로 지정
        nowdate = datetime.datetime.now()
        daytime = nowdate.strftime("%Y-%m-%d_%H%M%S")
        # 영상 저장 위치 설정
        video_name = 'ani/' + daytime + '.mp4'
        out_path = 'media/' + video_name
        # video codec 설정
        #fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc(*'AVC1')
        fps = 10.0
        # 현재 영상 프레임을 첫번째이미지로 설정(변경가능)
        frame = image_list[0][0]
        height, width, layers = frame.shape

        # video작성부분(저장위치, codec, fps, 영상프레임)
        video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        # 리스트에서 한 cut씩 가져옮
        for image in image_list:
            # 기본 5초에 이미지의 글자수를 10으로 나눈만큼 반복하여 같은 이미지 기록
            each_image_duration = 3 * 10 + int(image[1])
            for _ in range(each_image_duration):
                video.write(image[0])

        # 객체를 반드시 종료시켜주어야 한다
        video.release()

        # 영상 저장 위치 반환
        return video_name