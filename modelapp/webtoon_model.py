from tensorflow.python.keras.models import load_model
import cv2.cv2 as cv2
import numpy as np
from sklearn import preprocessing

class Web_Toon():
    IMAGE_SIZE = 224
    def plus_cut(self,img, polygons):
        cuts = []
        # opencv 비트 연산으로 이미지 합성
        for idx, polygon in enumerate(polygons):
            croped = img.copy()
            mask = np.zeros(croped.shape[:2], np.uint8)
            bg = np.ones_like(croped, np.uint8) * 255
            for poly in polygons[:idx + 1]:
                cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            cv2.bitwise_not(bg, bg, mask=mask)
            cut = bg + dst
            cuts.append(cut)
        return cuts


    # 컷 정렬 함수
    def sort_cut(self,padding_img_list, contours, read):
        height = padding_img_list[0].shape[0]
        width = padding_img_list[0].shape[1]
        n = 0
        sort_point = []
        for contour in contours:
            cx = contour.min(axis=0)[0][0] // int(width / 5)
            cy = contour.min(axis=0)[0][1] // int(height / 5)
            sort_point.append([cx, cy, n])
            n += 1
        if read == 'L':
            sort_point.sort(key=lambda x: (x[1], x[0]))
        elif read == 'R':
            sort_point.sort(key=lambda x: (x[1], -x[0]))
        sort_point = np.array(sort_point)
        if not contours:
            return []
        else:
            index = sort_point[:, 2].tolist()
            sort_contours = [contours[i] for i in index]
            return sort_contours


    # 모데 적용 함수
    def cut_model(self, img_list, model_name):
        img_input = []
        # 모델 입력 전처리
        for img in img_list:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_res = cv2.resize(img_gray, dsize=(self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            img_input.append(img_gray_res / 255)
        img_input = np.asarray(img_input)
        img_input = img_input.reshape(img_input.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        # 모델 적용
        model = load_model(model_name)
        predict = model.predict(img_input).reshape(len(img_input), self.IMAGE_SIZE, self.IMAGE_SIZE)
        # 출력 이미지 전처리
        labels = [np.around(label) * 255 for label in predict]
        labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
        return labels


    # 폴리곤, 컷 제작 함수
    def make_polygons(self, padding_img_list, labels, read):
        cuts_list = []
        polygons_list = []
        for idx, label in enumerate(labels):
            label = np.asarray(label, dtype=np.uint8)
            contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 컷 정렬
            contours = self.sort_cut(padding_img_list, contours, read)
            polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]
            cuts_list.append(self.plus_cut(padding_img_list[idx], polygons))
            polygons_list.append(polygons)
        return cuts_list, polygons_list


    # 말풍선 중심 함수
    def bubble_cent(self, cuts_polygons, bubble_polygons):
        bubble_cents = []
        for idx, page in enumerate(cuts_polygons):
            bubble_page_cent = []
            for cut in page:
                bubble_cut_cent = []
                for i in bubble_polygons[idx]:
                    M = cv2.moments(i)
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    if cv2.pointPolygonTest(cut, (cX, cY), True) >= 0:
                        bubble_cut_cent.append(i)
                bubble_page_cent.append(bubble_cut_cent)
            bubble_cents.append(bubble_page_cent)
        return bubble_cents


    def bubble_effect(self, cuts_list, bubbles_list, bubble_cents):
        scope_list = np.linspace(1, 1.6, 12)
        final_list = []
        for i, row_img in enumerate(cuts_list):
            page_img_list = []
            if not bubble_cents[i][0]:
                no_bubble = []
                for img in row_img:
                    no_bubble.append([img])
                final_list.append(no_bubble)
            else:
                for j, row_cut_img in enumerate(row_img):
                    bubble_img = bubbles_list[i][-1]
                    polygons = bubble_cents[i][j]
                    for polygon in polygons:
                        bubble_img_list = []
                        for scope in scope_list:
                            x, y, w, h = cv2.boundingRect(polygon)
                            roi_x = int(x + ((1 - scope) * w) * 0.5)
                            roi_y = int(y + ((1 - scope) * h) * 0.5)
                            roi = row_cut_img[roi_y: roi_y + int(scope * h), roi_x:roi_x + int(scope * w)].copy()
                            bubble_roi = bubble_img[y:y + h, x:x + w].copy()
                            res_bubble_roi = cv2.resize(bubble_roi, (roi.shape[1], roi.shape[0]))
                            pts = polygon - polygon.min(axis=0)
                            mask = np.zeros(bubble_roi.shape[:2], np.uint8)
                            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                            res_mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
                            res_mask_inv = cv2.bitwise_not(res_mask)
                            cut_bubble = cv2.bitwise_and(res_bubble_roi, res_bubble_roi, mask=res_mask)
                            cut_bg = cv2.bitwise_and(roi, roi, mask=res_mask_inv)
                            cut = cut_bg + cut_bubble
                            cut_img = row_cut_img.copy()
                            cut_img[roi_y: roi_y + int(scope * h), roi_x:roi_x + int(scope * w)] = cut
                            bubble_img_list.append(cut_img)
                        page_img_list.append(bubble_img_list)
                final_list.append(page_img_list)
        return final_list

    def bubble_len(self, cuts_list, bubbles_list, bubble_cents):
        bub_len_list = []
        for i, row_img in enumerate(cuts_list):
            page_len = []
            if not bubble_cents[i][0]:
                no_bubble = []
                for img in row_img:
                    no_bubble.append(0)
                bub_len_list.append(no_bubble)
            else:
                for j, row_cut_img in enumerate(row_img):
                    img_shape = row_cut_img.shape[0]
                    bubble_img = bubbles_list[i][-1]
                    polygons = bubble_cents[i][j]
                    if not polygons:
                        page_len.append(0)
                    else:
                        for polygon in polygons:
                            # 말풍선 영역 자르기
                            x, y, w, h = cv2.boundingRect(polygon)
                            bubble_roi = bubble_img[y:y + h, x:x + w].copy()
                            pts = polygon - polygon.min(axis=0)
                            # 마스크 설정
                            mask = np.zeros(bubble_roi.shape[:2], np.uint8)
                            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                            mask_inv = cv2.bitwise_not(mask)
                            # 마스크로 자르기
                            cut_bubble = cv2.bitwise_and(bubble_roi, bubble_roi, mask=mask)
                            gray_bubble = cv2.cvtColor(cut_bubble, cv2.COLOR_BGR2GRAY)
                            binarizer = preprocessing.Binarizer(threshold=150)
                            bub_binary = binarizer.transform(255 - (mask_inv + gray_bubble))
                            bub_len = np.sum(bub_binary) / img_shape
                            bub_len = np.round(bub_len, 2)
                            page_len.append(bub_len)
                bub_len_list.append(page_len)
        return bub_len_list

    def make_page_cut(self, img_list, read):
        cuts_list = []
        cut_labels = self.cut_model(img_list, 'model/best_gray_model_2.h5')
        bubble_labels = self.cut_model(img_list, 'model/bubble_gray_model.h5')
        hei = int(img_list[0].shape[0] * 0.15)
        wid = int(img_list[0].shape[1] * 0.15)
        padding_cut_labels = [cv2.copyMakeBorder(img, hei, hei, wid, wid, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img in
                              cut_labels]
        padding_bubble_labels = [cv2.copyMakeBorder(img, hei, hei, wid, wid, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img
                                 in bubble_labels]
        padding_img_list = [cv2.copyMakeBorder(img, hei, hei, wid, wid, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for img
                            in img_list]
        cuts_list, cuts_polygons = self.make_polygons(padding_img_list, padding_cut_labels, read)
        bubbles_list, bubble_polygons = self.make_polygons(padding_img_list, padding_bubble_labels, read)
        bubble_cents = self.bubble_cent(cuts_polygons, bubble_polygons)
        final_list = self.bubble_effect(cuts_list, bubbles_list, bubble_cents)
        bub_len_list = self.bubble_len(cuts_list, bubbles_list, bubble_cents)
        return final_list, bub_len_list

