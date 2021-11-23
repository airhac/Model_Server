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
                        #이미지 resize gkdutj 2500, 2500 중간 위치에 넣기
                        for k in j:
                            cols, rows, channel = k.shape
                            shape_list = [2300 / cols, 2300 / rows]
                            k = cv2.resize(k, (0, 0), fx=min(shape_list), fy=min(shape_list), interpolation=cv2.INTER_CUBIC)
                            cols, rows, channel = k.shape
                            space_width = int((wid - rows) / 2)
                            space_height = int((hei - cols) / 2)
                            back_image = np.zeros((hei, wid, 3), np.uint8)
                            back_image[space_height:space_height + cols, space_width:space_width + rows] = k
                            video.write(back_image)
                        #중간 이미지 60개 추가 -- 나중에 이미지 글자 픽셀값으로 변경 될 수 있음
                        for _ in range(60):
                            video.write(back_image)
                        #거꾸로 이미지 resize 하여서 2500, 2500 중간 위치에 넣기
                        for l in j[::-1]:
                            cols, rows, channel = l.shape
                            shape_list = [2300 / cols, 2300 / rows]
                            l = cv2.resize(1, (0,0) , fx=min(shape_list), fy=min(shape_list), interpolation=cv2.INTER_CUBIC)
                            cols, rows, channel = l.shape
                            space_width = int((wid - rows) / 2)
                            space_height = int((hei - cols) / 2)
                            back_image = np.zeros((hei, wid, 3), np.uint8)
                            back_image[space_height:space_height + cols, space_width:space_width + rows] = l
                            video.write(back_image)
                        last_frame = back_image
                    # 말풍선 효과 안넣었을 때
                    else:
                        #이미지 resize 하여서 2500, 2500 중간 위치에 넣기
                        image_hei, image_wid = j[0].shape[:2]
                        shape_list = [2300 / image_hei, 2300 / image_wid]
                        img_result = cv2.resize(j[0], (0, 0), fx=min(shape_list), fy=min(shape_list),
                                                interpolation=cv2.INTER_CUBIC)
                        back_image = np.zeros((hei, wid, 3), np.uint8)
                        cols, rows, channel = img_result.shape
                        space_width = int((wid - rows) / 2)
                        space_height = int((hei - cols) / 2)
                        # 말풍선 효과 길이만큼 보여주기- 나중에 이미지 글자 픽셀값으로 변경 될 수있음
                        each_image_duration = (len(j) * 2) + 60
                        for k in range(each_image_duration):
                            back_image[space_height:space_height + cols, space_width:space_width + rows] = img_result
                            video.write(back_image)
                        last_frame = back_image

                    # 여기서부터는 영상 전환 효과
                    # image_list가 마지막이 아니고 image가 마지막일경우
                    if (i + 1 == len(image)) and (idx + 1 < len(img_list)):
                        imag = img_list[idx + 1][0][0]
                    # image가 마지막이 아니고 같은 컷이 아닐경우
                    elif (i + 1 < len(image)) and (j[0].shape[:2] != image[i + 1][0].shape[:2]):
                        imag = image[i + 1][0]
                    else:
                        continue
                    # 효과로 보여줄 다음 이미지 resize
                    cols, rows, channel = imag.shape
                    shape_list = [2300 / cols, 2300 / rows]
                    imag = cv2.resize(imag, (0, 0), fx=min(shape_list), fy=min(shape_list),
                                        interpolation=cv2.INTER_CUBIC)
                    image_hei, image_wid = imag.shape[:2]
                    back_image = np.zeros((hei, wid, 3), np.uint8)
                    space_width = int((wid - image_wid) / 2)
                    space_height = int((hei - image_hei) / 2)
                    back_image[space_height:space_height + image_hei, space_width:space_width + image_wid] = imag



                    for p in range(1, int(fps + 1)):
                        frame = np.zeros((wid,hei, 3), dtype=np.uint8)
                        # 왼쪽으로...
                        if tran_effect == 'Lt':
                            dx = int((wid / fps) * p)
                            frame[:, 0:wid - dx, :] = last_frame[:, dx:wid, :]
                            frame[:, wid - dx:wid, :] = back_image[:, 0:dx, :]

                        # 오른쪽으로...
                        elif tran_effect == 'Rt':
                            dx = int((wid / fps) * p)
                            frame[:, 0:dx, :] = back_image[:, wid - dx:wid, :]
                            frame[:, dx:wid, :] = last_frame[:, 0:wid - dx, :]

                        # 위로...
                        elif tran_effect == 'U':
                            dx = int((hei / fps) * p)
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
            # 가장 첫번째 이미지 사이즈 받아서 사용
            hei, wid, channel = img_list[0][0][0].shape
            fps = 35.0
            video = cv2.VideoWriter(out_path, fourcc, fps, (wid, hei))
            # 맨 처음 하얀 화면 만들어서 비디오 기록
            frame = np.ones((hei, wid, 3), dtype=np.uint8) * 255
            last_frame = frame
            video.write(frame)
            # 3중 리스트로 되어있음
            for idx, image in enumerate(img_list):
                for i, j in enumerate(image):
                    # 컷 등장 디졸브 효과
                    for p in range(1, int(fps + 1)):
                        alpha = p / fps
                        frame = cv2.addWeighted(frame, 1 - alpha, j[0], alpha, 0)
                        video.write(frame)
                    # 말풍선 효과 넣었을 때
                    if ani_effect == 'B':
                        for k in j:
                            video.write(k)
                            imglast = k
                        for _ in range(60):
                            video.write(imglast)
                        for l in j[::-1]:
                            video.write(l)
                            la_img = l
                        last_frame = la_img
                    # 말풍선 효과 안넣었을 때
                    else:
                        each_image_duration = (len(j) * 2) + 60
                        for k in range(each_image_duration):
                            video.write(j[0])
                        last_frame = j[0]

        # 객체를 반드시 종료시켜주어야 한다
        # video.release()
                # 여기서부터는 영상 전환 효과
                # 마지막 이미지에는 효과 넣지 않기
                if idx + 1 >= len(img_list):
                    continue

                for p in range(1, int(fps + 1)):
                    frame = np.ones((hei, wid, 3), dtype=np.uint8) * 255
                    # 왼쪽으로...
                    if tran_effect == 'Lt':
                        dx = int((wid / fps) * p)

                        frame[:, 0:wid - dx, :] = last_frame[:, dx:, :]

                    # 오른쪽으로...
                    elif tran_effect == 'Rt':
                        dx = int((wid / fps) * p)
                        frame[:, dx:, :] = last_frame[:, 0:wid - dx, :]

                    # 위로...
                    elif tran_effect == 'U':
                        dx = int((hei / fps) * p)
                        frame[0:hei - dx, :, :] = last_frame[dx:, :, :]

                        # 디졸브 효과
                    elif tran_effect == 'D':
                        alpha = p / fps
                        frame = cv2.addWeighted(last_frame, 1 - alpha, frame, alpha, 0)

                    video.write(frame)
            # 객체를 반드시 종료시켜주어야 한다
            video.release()

        # 영상 저장 위치 반환
        return video_name