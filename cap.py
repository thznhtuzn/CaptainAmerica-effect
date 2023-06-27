import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
import cv2

#khai báo 1 số hàm của thư viện
mpose = mp.solutions.pose
pose = mpose.Pose()
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils
facedet = mp.solutions.face_mesh

#nhận video từ camera
video=cv2.VideoCapture(0)
fps = video

#khai báo các hàm cần thiết
def mask(frame, image, l_shoulder_x, l_shoulder_y):
    rol = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    rol = cv2.resize(rol, (250, 250), cv2.INTER_AREA)
    frame_h, frame_w, frame_c = frame.shape
    rol_h, rol_w, rol_c = rol.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for i in range(rol_h):
        for j in range(rol_w):
            if rol[i, j][3] != 0:
                if 0<=(l_shoulder_y - int(rol_h / 2)) + i < frame_h:
                        if 0<=(l_shoulder_x - int(rol_w / 2)) + j< frame_w:
                                frame[(l_shoulder_y - int(rol_h / 2)) + i, (l_shoulder_x - int(rol_w / 2)) + j] = rol[
                                    i, j]
    return frame

def position_data(lmlist):
    global wrist, index_mcp, index_tip, midle_mcp, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

def calculateDistance(p1,p2):
    x1,y1, x2,y2 = p1[0],p2[1],p2[0],p1[1]
    length = ((x2-x1)**2 + (y2-y1)**2)**(1.0/2)
    return length

def overlay_transparent(background_img, img_to_overlay_t,x,y,overlay_size=None):
    bg_img = background_img.copy()

    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(),overlay_size)

    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    mask = cv2.medianBlur(a,5)

    h,w, _ =overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img

#set kích thước video
size = (640,480)

l_shoulder_x = 0
l_shoulder_y = 0

#đọc các file hình ảnh
shield = cv2.imread("america.png",-1)
cap = cv2.imread("capt.png",-1)
anhnen_path = "anhBG.jpg"

#video segmentation
anh1_path = "anhnen.jpg"
anh2_path = "anhtrain.jpg"
anh1 = cv2.imread(anh1_path)
anh2 = cv2.imread(anh2_path)
anhnen = cv2.imread(anhnen_path)
anh1 = cv2.resize(anh1, size)
anh2 = cv2.resize(anh2, size)
anhnen = cv2.resize(anhnen, (640,480))
x = np.concatenate((anh1.reshape(-1, 3), anh2.reshape(-1, 3)), axis=0)
y = np.concatenate((np.zeros(anh1.shape[0] * anh1.shape[1]), np.ones(anh2.shape[0] * anh2.shape[1])))
class_weights = {0: 1, 1: 2} 
model = LogisticRegression(class_weight=class_weights)
model.fit(x, y)

#khai báo các cờ
check=0
flag=0

#xuất video
while True:
    ret,frame=video.read()

    #segmentation
    x_test = frame.reshape(-1, 3)
    y_pred = model.predict(x_test)
    
    output_image = np.where(y_pred.reshape(frame.shape[0], frame.shape[1], 1) == 0, anhnen, frame)
    frame2 = output_image.astype(np.uint8)

    img=cv2.flip(frame2,1)
    frame=cv2.flip(frame,1)
    img=cv2.resize(img,size)
    frame=cv2.resize(frame,size)
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    lmlist=[]
    falist=[]
    results = facedet.FaceMesh().process(imgRGB)

    #áp dụng mặt nạ
    if results.multi_face_landmarks:
        for a in results.multi_face_landmarks:
            for id,fm in enumerate(a.landmark):
                h,w,c = img.shape
                coorX, coorY = int (fm.x*w), int (fm.y*h)
                falist.append([coorX,coorY])
        fh = (falist[6][0],falist[6][1])
        nos = (falist[1][0],falist[1][1])
        ly = (falist[158][0],falist[159][1])
        ry = (falist[386][0],falist[386][1])
        facelen = calculateDistance(fh, nos)
        dist = calculateDistance(ly, ry)
        rat = dist/facelen

        #áp dụng mặt nạ, kích thước sẽ dựa vào khoảng cách 2 mắt
        if rat:
            nosX = fh[0]
            nosY = fh[1]
            cap_size = 3.5
            diam = round(dist*cap_size)
            x1 = round(nosX - (diam/2))
            y1 = round(nosY - (diam/2)-2)
            h,w,c = img.shape
            if x1<0:
                x1=0
            elif x1>w:
                x1=w
            if y1<0:
                y1=0
            elif y1>h:
                y1=h
            if x1+diam >w:
                diam = w -x1
            if y1+diam > h:
                diam = h-y1

            cap_size = diam,diam

            if (diam!=0):
                img = overlay_transparent(img, cap, x1, y1, cap_size)
    
    #áp dụng khiên captian vào tay
    if result.multi_hand_landmarks:
        h,w,c = img.shape
        for handslms in result.multi_hand_landmarks:
            for id, lm in enumerate(handslms.landmark):
                
                coorx, coory = int(lm.x*w), int(lm.y*h)
                lmlist.append([coorx,coory])

        position_data(lmlist)
        palm=calculateDistance(wrist, index_mcp)
        distance = calculateDistance(index_tip, pinky_tip)
        ratio = distance/palm

        #kiểm tra khoảng cách từ ngón tay đến 1 điểm cho trước, 1 điểm đỏ trên màn hình
        check=calculateDistance(index_tip, (size[0],size[1]/2))

        #áp dụng khiên vào tay, kích thước dựa vào khoảng cách từ ngón giữa đến cổ tay
        if ratio<.9:
            centerX=midle_mcp[0]
            centerY=midle_mcp[1]
            shield_size=5.0
            diameter=round(palm*shield_size)
            x1=round(centerX-(diameter/2))
            y1=round(centerY-(diameter/2))
            h,w,c = img.shape
            if x1<0:
                x1=0
            elif x1>w:
                x1=w
            if y1<0:
                y1=0
            elif y1>h:
                y1=h
            if x1+diameter >w:
                diameter = w -x1
            if y1+diameter > h:
                diameter = h-y1
            shield_size = diameter,diameter
            if (diameter != 0):
                img = overlay_transparent(img, shield, x1, y1, shield_size)

    if  (flag==1):
        pass
    else:
        if (0< check < 200):
            flag=1

    #khi chưa đạt đủ khoảng cách check thì show video bình thường
    if (flag==0):
        cv2.circle(frame, (size[0],240), 5, (0,0,255), -1)
        cv2.imshow("Frame",frame)

    #khi đủ check thì flag = 1, show video hiệu ứng
    else:
        cv2.circle(img, (size[0],240), 5, (0,0,255), -1)
        cv2.imshow("Frame",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()