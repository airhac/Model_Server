import cv2.cv2 as cv2
import numpy as np

class ComicFrameBook():

    def __init__(self, ani_effect, bubbles, cuts, cut_polygons, bubble_centers, page_len, padding=300):
        self.ani_effect = ani_effect
        self.cuts = cuts
        self.bubbles = bubbles
        self.cut_polygons = cut_polygons
        self.bubble_centers = bubble_centers

        self.page_len = page_len
        self.padding = padding  # 컷 분리 때 사용하는 패딩과 같은 값이어야 합니다.

        self.page_count = 0  # 결과 출력용도

        for i in range(self.page_len):
            # 비어있는 풍선의 경우
            if len(self.bubbles[i]) == 0:
                continue

            self.matching_bubble2cut(self.cuts[i], self.bubbles[i], self.bubble_centers[i], self.cut_polygons[i])

    # 이건 단일 페이지에 작동합니다.
    def matching_bubble2cut(self, cuts, bubbles, bubble_centers, polygons, ):
        for i, bubble_center in enumerate(bubble_centers):
            for j, polygon in enumerate(polygons):

                # 말풍선의 중심점이 각 컷의 폴리곤 내부에 있는지 확인합니다.
                is_inpolygon = cv2.pointPolygonTest(polygon, bubble_center[:2], True)
                if is_inpolygon >= 0:
                    bubbles[i]['matching_cut_num'] = j
                    cuts[j]['matching_bub_num'].append(i)  # 클래스를 여러 번 선언하면 이 리스트가 중복되는 이슈가 있습니다.
                    break  # matching_bub_num  은 한 컷에 여러 풍선이 매칭되는 경우를 위해 리스트로 되어있습니다.
                    # 그런데 해당 리스트가 초기화 되는 부분은 컷 분리 함수이고 클래스와 동시에 동작하지 않습니다.
                    # 때문에 깊은복사 이슈가 생겨서 리스트가 중복됩니다. 아직 수정하지 못했습니다.
                    # 그래서 현재는 클래스를 다시 선언하시려면 컷 분리 부분도 다시 실행해주셔야합니다.

    # 매칭된 컷-버블 정보를 이용해 make_bubblescope_cut 함수를 실행합니다.
    def makeframe_proc(self):
        frame_pages = []
        self.page_count = 0
        for page_num in range(self.page_len):  # 페이지단위로 움직입니다.
            frames = []
            for idx, cut in enumerate(self.cuts[page_num]):
                bub_nums = cut['matching_bub_num']  # 각 컷에 매칭되는 말풍선을 찾습니다.

                ############################
                # 말풍선이 없는 경우. 지금은 이미지 하나만 그대로 들어갑니다. 나중에 개수를 수정할 필요가 있습니다.
                if len(bub_nums) == 0:
                    frames.append([cut['image']])

                    self.page_count += 1
                    continue

                # 말풍선 확대와 복사 작업을 실행합니다.
                for bub_num in bub_nums:
                    target_bub = self.bubbles[page_num][bub_num]
                    bub_centroid = self.bubble_centers[page_num][bub_num]
                    frames.append(self.make_bubblescope_cut(self.ani_effect, target_bub, cut, bub_centroid, self.padding))

            frame_pages.append(frames)

        return frame_pages

    # 컷 한장, 버블 한장당 작동합니다.
    # 확대 배율은 현재 임의로 설정 해 두었습니다.
    def make_bubblescope_cut(self, ani_effect, bubble, cut, bub_centroid, padding):
        half_padding = int(padding / 2)
        if ani_effect == "B":
            scope_list = np.linspace(1, 1.6, 12)
        else:
            scope_list = np.linspace(1, 1, 12)

        cut_bg = cut['image']
        bub_fg = bubble['image']
        cut_xywh = cut['xywh']
        bub_xywh = bubble['xywh']

        # mask 와 mask_inv 를 만들자
        mask = bubble['mask']

        bubble_scope_cuts = []

        for scope in scope_list:
            # 강조를 위해서 resize 합니다.
            # 리사이징 때문에 mask_inv 는 여기서 만듭니다.
            resize_scale = [int(bub_xywh[2] * scope), int(bub_xywh[3] * scope)]

            bub_fg_resize = cv2.resize(bub_fg, dsize=resize_scale, interpolation=cv2.INTER_AREA)

            mask_resize = cv2.resize(mask, dsize=resize_scale, interpolation=cv2.INTER_AREA)
            mask_inv = cv2.bitwise_not(mask_resize)

            # 이미지 변경할 시작점을 잡아줍니다.
            bubble_start_h = bub_centroid[1] - cut_xywh[1] - int(mask_resize.shape[0] / 2) + half_padding
            bubble_start_w = bub_centroid[0] - cut_xywh[0] - int(mask_resize.shape[1] / 2) + half_padding

            # 원본 이미지에서 배경이 될 부분을 뽑습니다.
            roi = cut_bg[bubble_start_h:bubble_start_h + mask_resize.shape[0], bubble_start_w:bubble_start_w + mask_resize.shape[1]]

            # 마스크 이용해서 오려내기
            masked_fg = cv2.bitwise_and(bub_fg_resize, bub_fg_resize, mask=mask_resize)
            masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # 이미지 합성
            added = masked_fg + masked_bg
            cut_bg_copy = cut_bg.copy()
            cut_bg_copy[bubble_start_h:bubble_start_h + mask_resize.shape[0], bubble_start_w:bubble_start_w + mask_resize.shape[1]] = added

            bubble_scope_cuts.append(cut_bg_copy)

            self.page_count += 1

        return bubble_scope_cuts