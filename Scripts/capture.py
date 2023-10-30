#capture.py

import cv2

#顔検出器を初期化
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#映像を読み込むオブジェクト
cap= cv2.VideoCapture(0) #0はデフォルトのカメラデバイス

ret,frame = cap.read()
#グレイスケール化
faces = face_cascade.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

if len(faces) > 0:
    (x,y,w,h) = faces[0] #顔検出器によって返される位置情報を変数に取る
    track_window = (x,y,w,h) #トラッキングウィンドウの位置とサイズ
    roi = frame[y:y+h, x:x+w] #region of interest. 顔がある部分の領域
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #BGR色空間からHSV色空間に変換
    mask = cv2.inRange(hsv_roi,(0,60,32),(180,255,255)) #指定範囲内の色は白に、それ以外は黒に変換
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) #指定された色相のヒストグラムを計算
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX) #ヒストグラムの正規化とスケーリング
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) #トラッキング処理（後述）の停止基準。10回繰り返したら終了

while True:
    #カメラからフレームを取得
    ret, frame = cap.read()

    #フレームの取得に成功したら
    if ret:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) #ヒストグラムの逆投影を行う
        ret, track_window=cv2.meanShift(dst,track_window,term_crit) #トラッキングを行う
        x,y,w,h=track_window
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        #フレームをウィンドウで表示
        cv2.imshow('Camera',frame)
    #qを押したらループ解除
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#終了動作
cap.release()
cv2.destroyAllWindows()