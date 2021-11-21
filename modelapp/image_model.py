import cv2.cv2 as cv2
from tensorflow.python.keras.models import load_model
import numpy as np

class Image_Model():
    # 이미지 별 사이즈 적용 수정 필요
    # 컷 분리 함수
    def split_cut(self, img, polygon):
        x, y, w, h = cv2.boundingRect(polygon)
        croped = img[y:y + h, x:x + w].copy()
        pts = polygon - polygon.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        cut = bg + dst

        # 하얀 배경에 중간위치에 cut이미지 넣어서 저장해줌
        height = 650
        width = 450
        back_image = np.ones((height, width, 3), np.uint8) * 255
        cols, rows, channel = cut.shape
        space_height = int((height - cols) / 2)
        space_width = int((width - rows) / 2)
        back_image[space_height:space_height + cols, space_width:space_width + rows] = cut

        return back_image

    def sort_cut(self,contours):
        n = 0
        centroids = []
        for contour in contours:
            centroid = cv2.moments(contour)
            cx = int(centroid['m10'] / centroid['m00'])
            cy = int(centroid['m01'] / centroid['m00'])
            centroids.append([cx, cy, n])
            n += 1
        centroids.sort(key=lambda x: (x[1], x[0]))
        centroids = np.array(centroids)
        index = centroids[:, 2].tolist()
        sort_contours = [contours[i] for i in index]
        return sort_contours

    # model 함수..
    def make_cut(self, img_list):
        IMAGE_SIZE = 224
        model = load_model('model/best_gray_model_2.h5')
        img_input = []
        cuts = []
        # 모델 입력 전처리
        img_gray = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2GRAY)
        for img in img_list:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_res = cv2.resize(img_gray, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            img_input.append(img_gray_res / 255)
        img_input = np.asarray(img_input)
        img_input = img_input.reshape(img_input.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
        # 모델 적용
        img_predict = model.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)
        # 출력 이미지 전처리
        labels = [np.around(label) * 255 for label in img_predict]
        labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
        for idx, label in enumerate(labels):
            label = np.asarray(label, dtype=np.uint8)
            contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 컷 정렬
            contours = self.sort_cut(contours)
            background = np.full(label.shape, 255, dtype=np.uint8)
            polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]
            for polygon in polygons:
                cuts.append(self.split_cut(img_list[idx], polygon))
        return cuts