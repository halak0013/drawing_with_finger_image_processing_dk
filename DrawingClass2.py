#Bismillahirrahmanirrahim
import cv2
import numpy as np
import cvzone
import cvzone.HandTrackingModule as htm
from dk_connection import ImageReceiver, SocketServer
import time
# daha fazla renk siyah, kahverengi, turuncu, sarı, mor, pembe, beyaz
# sitcker
# seçimin ne olduğu

class HandDrawingApp:
    def __init__(self, target_width=1920, target_height=1080, detection_confidence=0.40, max_hands=1, source="camera"):
        self.target_width = target_width
        self.target_height = target_height
        self.detection_confidence = detection_confidence
        self.max_hands = max_hands
        self.canvas_mask = np.zeros(
            (self.target_height, self.target_width, 3), np.uint8)

        self.xp = 0
        self.yp = 0
        self.first_time = True
        self.clear_time = 0.0

        self.source = source

        if source == "camera":
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, self.target_width)
            self.cap.set(4, self.target_height)
        else:
            self.sock_server = SocketServer()
            self.image_receiver = ImageReceiver(self.sock_server)

        self.header = cv2.imread(
            "Header/header.png", cv2.IMREAD_UNCHANGED)
        
        self.img_width = self.header.shape[1]

        self.brushes = {
            # BGR formatında kırmızı
            0: {"name": "red", "color": (39, 39, 227), "img": self.header},
            # BGR formatında yeşil
            1: {"name": "green", "color": (39, 227, 72), "img": self.header},
            # BGR formatında mavi
            2: {"name": "blue", "color": (227, 98, 39), "img": self.header},
            #255, 236, 6
            3: {"name": "yelow", "color": (6, 236, 255), "img": self.header},
            #255, 149, 6
            4: {"name": "orange", "color": (6, 149, 255), "img": self.header},
            #255, 224, 136
            5: {"name": "skin", "color": (136, 224, 255), "img": self.header},
            #141, 73, 0
            6: {"name": "brown", "color": (0, 73, 141), "img": self.header},
            7: {"name": "white", "color": (255, 255, 255), "img": self.header},
            8: {"name": "black", "color": (10, 10, 10), "img": self.header},
            9: {"name": "rubber", "color": (0, 0, 0), "img": self.header}
        }

        self.brush_index = 0

        self.resize_images()

        # Initialize hand detector
        self.detector = htm.HandDetector(
            detectionCon=self.detection_confidence, maxHands=self.max_hands)
        self.sock_server = SocketServer()
        self.image_receiver = ImageReceiver(self.sock_server)

    def resize_images(self):
        for i in range(len(self.brushes)):
            self.brushes[i]["img"] = cv2.resize(
                self.brushes[i]["img"], (self.target_width, 100))

    def is_selection_mode(self, hands):
        """
        1 çizim modu\n
        2 seçim modu
        """
        finger_up = self.detector.fingersUp(hands[0])
        mode = ""
        if finger_up[1] and finger_up[2]:
            mode = "selection"
        else:
            mode = "drawing"
        return mode

    def finger_position(self, hands):
        return hands[0]["lmList"][8]
    def clear_all(self, hands, img):
        finger_up = self.detector.fingersUp(hands[0])
        if finger_up[0] and finger_up[1] and finger_up[2] and finger_up[3] and finger_up[4]:
            self.clear_time+=1
            cv2.putText(img, str(10 - self.clear_time), (self.target_width//2, 150),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 200, 0), 1)
            if self.clear_time > 10:
                self.canvas_mask = np.zeros(
                    (self.target_height, self.target_width, 3), np.uint8)
                self.clear_time=0
                
        else:
            self.clear_time=0

    def draw(self, hands, img: cv2.typing.MatLike):
        if hands:
            x, y, z = self.finger_position(hands)
            cv2.circle(img, (x, y),
                       10, self.brushes[self.brush_index]["color"], cv2.FILLED)
            if self.xp == 0 and self.yp == 0 or self.first_time:
                self.xp, self.yp = x, y
            if self.brush_index == 3:
                cv2.line(self.canvas_mask, (self.xp, self.yp),
                         (x, y), self.brushes[self.brush_index]["color"], 30)
            else:
                cv2.line(self.canvas_mask, (self.xp, self.yp),
                         (x, y), self.brushes[self.brush_index]["color"], 13)
            self.xp, self.yp = x, y

    def masking(self, img):
        # Create a mask where all non-black pixels are white and black pixels are black
        # Convert to grayscale
        mask = cv2.cvtColor(self.canvas_mask, cv2.COLOR_BGR2GRAY)
        # Create a binary mask
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Invert the mask to keep the drawn areas
        inverted_mask = cv2.bitwise_not(mask)

        # Keep only the drawn areas of img
        img = cv2.bitwise_and(img, img, mask=inverted_mask)
        # Add the canvas_mask to the original image
        img = cv2.add(img, self.canvas_mask)
        return img

    def selecting(self, finger_pos):
        x = self.target_width / 100
        is_y = finger_pos[1] < 100

        if 17 * x < finger_pos[0] < 23.5 * x and is_y:
            self.brush_index = 0
        elif 26 * x < finger_pos[0] < 32.5 * x and is_y:
            self.brush_index = 1
        elif 35 * x < finger_pos[0] < 41 * x and is_y:
            self.brush_index = 2
        elif 42.5 * x < finger_pos[0] < 50 * x and is_y:
            self.brush_index = 3
        elif 52 * x < finger_pos[0] < 58 * x and is_y:
            self.brush_index = 4
        elif 60.5 * x < finger_pos[0] < 67 * x and is_y:
            self.brush_index = 5
        elif 67 * x < finger_pos[0] < 73.5 * x and is_y:
            self.brush_index = 6
        elif 74.5 * x < finger_pos[0] < 81 * x and is_y:
            self.brush_index = 7
        elif 81 * x < finger_pos[0] < 88 * x and is_y:
            self.brush_index = 8
        elif 92.5 * x < finger_pos[0] < 96.5 * x and is_y:
            self.brush_index = 9

    def get_img(self):
        img = None
        success = None
        if self.source == "sock":
            img = self.image_receiver.get_image()
            success = img is not None
        else:
            success, img = self.cap.read()
        return success, img

    def run(self):
        while True:
            success, img = self.get_img()
            if not success:
                print(self.source)
                continue
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (self.target_width, self.target_height))
            # Detect hands
            hands, img = self.detector.findHands(img)

            # Overlay header
            img = cvzone.overlayPNG(
                img, self.brushes[self.brush_index]["img"], [0, 0])

            if hands:
                cv2.rectangle(img, (0, 115), (self.target_width, 100),
                              self.brushes[self.brush_index]["color"], cv2.FILLED)
                self.clear_all(hands,img)
                if self.is_selection_mode(hands) == "drawing":
                    self.draw(hands,img)
                    self.first_time = False
                elif self.is_selection_mode(hands) == "selection":
                    self.selecting(self.finger_position(hands))
                    self.first_time = True
                else:
                    self.first_time = True
            img = self.masking(img)

            # Overlay header
            img = cvzone.overlayPNG(
                img, self.brushes[self.brush_index]["img"], [0, 0])
            cv2.imshow("Deneyap Ciz", img)
            # cv2.imshow("mask", self.canvas_mask)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                print("Çık")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandDrawingApp(source="camera")
    app.run()
