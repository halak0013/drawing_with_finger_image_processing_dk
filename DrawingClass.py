#Bismillahirrahmanirrahim
import cv2
import numpy as np
import cvzone
import cvzone.HandTrackingModule as htm
from dk_connection import ImageReceiver, SocketServer

# daha fazla renk siyah, kahverengi, turuncu, sarı, mor, pembe, beyaz
# sitcker
# seçimin ne olduğu
# 
class HandDrawingApp:
    def __init__(self, target_width=1800, target_height=1000, detection_confidence=0.65, max_hands=1):
        self.target_width = target_width
        self.target_height = target_height
        self.detection_confidence = detection_confidence
        self.max_hands = max_hands
        self.canvas_mask = np.zeros(
            (self.target_height, self.target_width, 3), np.uint8)

        self.xp = 0
        self.yp = 0
        self.first_time = True

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.target_width)
        self.cap.set(4, self.target_height)

        # Load headers
        self.header_blue = cv2.imread(
            "Header/top_bar_blue.png", cv2.IMREAD_UNCHANGED)

        self.header_green = cv2.imread(
            "Header/top_bar_green.png", cv2.IMREAD_UNCHANGED)
        self.header_red = cv2.imread(
            "Header/top_bar_red.png", cv2.IMREAD_UNCHANGED)
        self.header_rubber = cv2.imread(
            "Header/top_bar_rubber.png", cv2.IMREAD_UNCHANGED)
        self.img_width = self.header_blue.shape[1]

        self.brushes = {
            # BGR formatında kırmızı
            0: {"name": "red", "color": (39, 39, 227), "img": self.header_red},
            # BGR formatında yeşil
            1: {"name": "green", "color": (39, 227, 72), "img": self.header_green},
            # BGR formatında mavi
            2: {"name": "blue", "color": (227, 98, 39), "img": self.header_blue},
            # Silgi için siyah
            3: {"name": "rubber", "color": (0, 0, 0), "img": self.header_rubber}
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

    def is_sellection_mode(self, hands):
        """
        1 çizim modu\n
        2 seçim modu
        """
        finger_up = self.detector.fingersUp(hands[0])
        mode = ""
        if finger_up[1] and finger_up[2]:
            mode = "sellection"
        elif finger_up[1]:
            mode = "drawing"
        else:
            mode = ""
        return mode

    def finger_position(self, hands):
        return hands[0]["lmList"][8]
    def clear_all(self, hands):
        finger_up = self.detector.fingersUp(hands[0])
        if finger_up[0] and finger_up[1] and finger_up[2] and finger_up[3] and finger_up[4]:
            self.canvas_mask = np.zeros(
                (self.target_height, self.target_width, 3), np.uint8)

    def draw(self, hands):
        if hands:
            x, y, z = self.finger_position(hands)
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

    def sellecting(self, finger_pos):
        x = self.target_width / 100
        if 24 * x < finger_pos[0] < 31 * x and finger_pos[1] < 100:
            self.brush_index = 0
        if 36 * x < finger_pos[0] < 44 * x and finger_pos[1] < 100:
            self.brush_index = 1
        if 49 * x < finger_pos[0] < 56 * x and finger_pos[1] < 100:
            self.brush_index = 2
        if 66 * x < finger_pos[0] < 71 * x and finger_pos[1] < 100:
            self.brush_index = 3

    def get_img(self, source="camera"):
        img = None
        succes = None
        if source == "sock":
            img = self.image_receiver.get_image()
            succes = img is not None
        else:
            succes, img = self.cap.read()
        return succes, img

    def run(self):
        while True:
            success, img = self.get_img("sock")
            if not success:
                continue
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (self.target_width, self.target_height))
            # Detect hands
            hands, img = self.detector.findHands(img)

            # Overlay header
            img = cvzone.overlayPNG(
                img, self.brushes[self.brush_index]["img"], [0, 0])

            if hands:
                if self.is_sellection_mode(hands) == "drawing":
                    self.draw(hands)
                    self.first_time = False
                elif self.is_sellection_mode(hands) == "sellection":
                    self.sellecting(self.finger_position(hands))
                    self.first_time = True
                else:
                    self.first_time = True
                self.clear_all(hands)
            img = self.masking(img)

            # Overlay header
            img = cvzone.overlayPNG(
                img, self.brushes[self.brush_index]["img"], [0, 0])
            cv2.imshow("Deneyap", img)
            # cv2.imshow("mask", self.canvas_mask)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                print("Çık")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
