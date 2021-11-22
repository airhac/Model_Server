import cv2.cv2 as cv2
from easyocr import easyocr
import datetime
import numpy as np

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

    # [이미지,글자수]의 리스트를 받아 영상으로 만들고 저장하는 함수
    def new_view_seconds(self, img_list, t_c, ani_effect, tran_effect):
        # 영상이름 오늘 날자와 시간으로 지정
        nowdate = datetime.datetime.now()
        daytime = nowdate.strftime("%Y-%m-%d_%H%M%S")
        # 영상 저장 위치 설정
        video_name = 'ani/' + daytime + '.mp4'
        out_path = 'media/' + video_name
        # video codec 설정
        fourcc = cv2.VideoWriter_fourcc(*'AVC1')

        # 여기서부터 컷과 책 형태 분리 하기
        # toon 형식
        if t_c == 'T':
            wid, hei = 2500, 2500
            fps = 35.0
            video = cv2.VideoWriter(out_path, fourcc, fps, (wid, hei))
            back_image = np.zeros((hei, wid, 3), np.uint8)
            # 3중 리스트로 되어있음
            for idx, image in enumerate(img_list):
                for i, j in enumerate(image):
                    # 말풍선 효과 넣었을 때
                    if ani_effect == 'B':
                        for k in j:
                            cols, rows, channel = k.shape
                            space_width = int((wid - rows) / 2)
                            space_height = int((hei - cols) / 2)
                            back_image = np.zeros((hei, wid, 3), np.uint8)
                            back_image[space_height:space_height + cols, space_width:space_width + rows] = k
                            video.write(back_image)
                        for _ in range(60):
                            video.write(back_image)
                        for l in j[::-1]:
                            cols, rows, channel = l.shape
                            space_width = int((wid - rows) / 2)
                            space_height = int((hei - cols) / 2)
                            back_image = np.zeros((hei, wid, 3), np.uint8)
                            back_image[space_height:space_height + cols, space_width:space_width + rows] = l
                            video.write(back_image)
                        last_frame = back_image
                    # 말풍선 효과 안넣었을 때
                    else:
                        image_hei, image_wid = j[0].shape[:2]
                        img_result = cv2.resize(j[0], (2300, int((2300 / image_wid) * image_hei)),
                                                interpolation=cv2.INTER_CUBIC)
                        back_image = np.zeros((hei, wid, 3), np.uint8)
                        cols, rows, channel = img_result.shape
                        space_width = int((wid - rows) / 2)
                        each_image_duration = (len(j) * 2) + 60
                        for k in range(each_image_duration):
                            if cols < hei:
                                space_height = int((hei - cols) / 2)
                                back_image[space_height:space_height + cols,
                                space_width:space_width + rows] = img_result
                            else:
                                nx = int(((cols - 2500) / each_image_duration) * k)
                                back_image[:, space_width:space_width + rows] = img_result[nx:nx + 2500, :]
                            video.write(back_image)
                        last_frame = back_image

                    # 여기서부터는 영상 전환 효과
                    if (i + 1 == len(image)) and (idx + 1 < len(img_list)):
                        imag = img_list[idx + 1][0][0]
                    elif (i + 1 < len(image)) and (j[0].shape[:2] != image[i + 1][0].shape[:2]):
                        imag = image[i + 1][0]
                    else:
                        continue
                    image_hei, image_wid = imag.shape[:2]

                    # 말풍선 효과 넣었을 때
                    if ani_effect == 'B':
                        back_image = np.zeros((hei, wid, 3), np.uint8)
                        space_width = int((wid - image_wid) / 2)
                        space_height = int((hei - image_hei) / 2)
                        back_image[space_height:space_height + image_hei, space_width:space_width + image_wid] = imag

                    # 말풍선 효과 안넣었을 때
                    else:
                        img_result = cv2.resize(imag, (2300, int((2300 / image_wid) * image_hei)),
                                                interpolation=cv2.INTER_CUBIC)
                        back_image = np.zeros((hei, wid, 3), np.uint8)
                        cols, rows, channel = img_result.shape
                        space_width = int((wid - rows) / 2)
                        if cols < hei:
                            space_height = int((hei - cols) / 2)
                            back_image[space_height:space_height + cols, space_width:space_width + rows] = img_result
                        else:
                            back_image[:, space_width:space_width + rows] = img_result[0:2500, :]

                    for p in range(1, int(fps + 1)):
                        frame = np.zeros((hei, wid, 3), np.uint8)
                        # 왼쪽으로...
                        if tran_effect == 'Lt':
                            dx = int((wid / fps) * p)
                            frame = np.zeros((hei, wid, 3), dtype=np.uint8)
                            frame[:, 0:wid - dx, :] = last_frame[:, dx:wid, :]
                            frame[:, wid - dx:wid, :] = back_image[:, 0:dx, :]

                        # 오른쪽으로...
                        elif tran_effect == 'Rt':
                            dx = int((wid / fps) * p)
                            frame = np.zeros((hei, wid, 3), dtype=np.uint8)
                            frame[:, 0:dx, :] = back_image[:, wid - dx:wid, :]
                            frame[:, dx:wid, :] = last_frame[:, 0:wid - dx, :]

                        # 위로...
                        elif tran_effect == 'U':
                            dx = int((hei / fps) * p)
                            frame = np.zeros((hei, wid, 3), dtype=np.uint8)
                            frame[0:hei - dx, :, :] = last_frame[dx:hei, :, :]
                            frame[hei - dx:hei, :, :] = back_image[0:dx, :, :]

                        # 디졸브 효과
                        elif tran_effect == 'D':
                            alpha = p / fps
                            frame = cv2.addWeighted(last_frame, 1 - alpha, back_image, alpha, 0)

                        video.write(frame)
            # 객체를 반드시 종료시켜주어야 한다
            video.release()
        # comic 만화책 형식일때
        else:
            pass

        # 객체를 반드시 종료시켜주어야 한다
        # video.release()

        # 영상 저장 위치 반환
        return video_name