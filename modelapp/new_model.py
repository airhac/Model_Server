from tensorflow.python.keras.models import load_model
import cv2.cv2 as cv2
import numpy as np

class New_Model():
    IMAGE_SIZE = 224
    # 컷 모델
    model = load_model('model/best_gray_model_2.h5')
    # 말풍선 모델
    model_b = load_model('model/bubble_gray_model.h5')
    # 말풍선 모델
    def image_preproc(self,image_list):  # 이 코드는 전처리부분만을 가져왔습니다.
        img_input = []

        # 모델 입력 전처리
        for img in image_list:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_res = cv2.resize(img_gray, dsize=(self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            img_input.append(img_gray_res / 255)
        img_input = np.asarray(img_input)
        img_input = img_input.reshape(img_input.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

        # 모델 적용 컷 or 말풍선
        cut_predict = self.model.predict(img_input).reshape(len(img_input), self.IMAGE_SIZE, self.IMAGE_SIZE)
        bubble_predict = self.model_b.predict(img_input).reshape(len(img_input), self.IMAGE_SIZE, self.IMAGE_SIZE)

        # 출력 이미지 전처리
        # 이제 컷과 말풍선을 동시에 처리합니다.
        labels_list = []
        for predict in [cut_predict, bubble_predict]:
            labels = [np.around(label) * 255 for label in predict]
            # 여기서 다시 출력이미지 사이즈를 키움
            labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
            labels_list.append(labels)

        return labels_list[0], labels_list[1]  # 0 : cut, 1 : bubble


    def split_cut_b(self,img, polygon, page_num, is_bubble=False):
        x, y, w, h = cv2.boundingRect(polygon)  # 폴리곤으로 bounding박스 그림
        croped = img[y:y + h, x:x + w].copy()  # 원본 이미지에서 자름.

        # 패딩을 추가해보자
        # 컷일 경우에만 패딩을 추가합니다.
        if not is_bubble:
            # ******* 저는 작은 이미지로 해서 패딩을 100정도 줬습니다.
            # ******* 큰 이미지로 하시게 되면 패딩을 더 넣어주셔야 합니다.
            padding = 500
            startline = int(padding / 2)

            img_padding = np.zeros([h + padding, w + padding, 3], np.uint8)
            img_padding[startline:startline + h, startline: startline + w] = croped

            croped = img_padding

            pts = polygon - polygon.min(axis=0) + startline
        else:
            pts = polygon - polygon.min(axis=0)

        mask = np.zeros([croped.shape[0], croped.shape[1]], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        dst = cv2.bitwise_and(croped, croped, mask=mask)  # 빈 공간에 컷만 추가.

        # 변경점. 이제 모든 이미지는 x, y, w, h, mask, matching_num 값을 포함합니다.
        # 컷에 말풍선이 없거나 말풍선이 잘못 검출 된 경우를 위해 여기서 초기화합니다.
        if is_bubble:
            return {'image': dst, 'xywh': [x, y, w, h], 'mask': mask, 'matching_cut_num': -1}
        else:
            return {'image': dst, 'xywh': [x, y, w, h], 'mask': mask, 'matching_bub_num': []}


    def sort_cut_b(self,img_list, contours, read, is_bubble=False):
        height = img_list[0].shape[0]
        width = img_list[0].shape[1]
        n = 0
        centroids = []
        bubble_centers = []

        for contour in contours:
            cx = contour.min(axis=0)[0][0] // int(width / 5)  # 이것은 순서를 정하기 위해서 뽑습니다.
            cy = contour.min(axis=0)[0][1] // int(height / 5)

            centroids.append([cx, cy, n])
            n += 1

        if read == 'L':
            centroids.sort(key=lambda x: (x[1], x[0]))
        elif read == 'R':
            centroids.sort(key=lambda x: (x[1], -x[0]))
        centroids = np.array(centroids)

        index = centroids[:, 2].tolist()
        sort_contours = [contours[i] for i in index]

        if is_bubble:
            for contour in sort_contours:
                centroid = cv2.moments(contour)
                bx = int(centroid['m10'] / centroid['m00'])  # 이건 말풍선과 컷을 매칭하기 위해서 뽑습니다.
                by = int(centroid['m01'] / centroid['m00'])

                bubble_centers.append([bx, by, n])

            return sort_contours, centroids, bubble_centers  # 이걸 이용해 컷과 매칭합니다.

        return sort_contours, centroids


    def make_cut_bubble(self, img_list, labels, read, is_bubble=False):
        cuts_list = []
        centroids_list = []
        polygons_list = []
        bubble_centers_list = []

        for idx, label in enumerate(labels):
            cuts = []
            label = np.asarray(label, dtype=np.uint8)

            contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # countour를 찾음.

            # 컷에 말풍선이 하나도 없을 때의 탈출조건
            if is_bubble and (len(contours) == 0):
                cuts_list.append([])
                bubble_centers_list.append([])
                continue

            # 아직 잘못 나온 이미지 처리 안함..
            # print("여기")
            # print(np.shape(contours))
            # print(contours[0].shape)
            # for cont in contours:
            #     if cont[0] < 200:
            #         contours.remove(cont)

            # 컷 정렬
            if is_bubble:
                contours, centroids, bubble_centers = self.sort_cut_b(img_list, contours, read, is_bubble=True)
            else:
                contours, centroids = self.sort_cut_b(img_list, contours, read)

            polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]  # contour reshape

            if is_bubble:
                for polygon in polygons:
                    cuts.append(self.split_cut_b(img_list[idx], polygon, page_num=idx, is_bubble=True))

                bubble_centers_list.append(bubble_centers)
            else:
                for polygon in polygons:
                    cuts.append(self.split_cut_b(img_list[idx], polygon, page_num=idx))

                polygons_list.append(polygons)

            cuts_list.append(cuts)
            centroids_list.append(centroids)

        # 변경점. 이제 모든 출력값은 페이지마다로 구분됩니다.
        # [0] : 페이지0번, [0][0] : 페이지0번의 0번 째 컷 or 버블의 데이터
        if is_bubble:
            return cuts_list, centroids_list, bubble_centers_list  # bubble_centers 는 말풍선 컷 매칭을 위해 뽑습니다.
        else:
            return cuts_list, centroids_list, polygons_list  # 폴리곤은 말풍선 위치결정을 위해서 뽑습니다.
