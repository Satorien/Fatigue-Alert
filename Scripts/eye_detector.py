from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import playsound
import time

# 顔検出器の初期化
face_detector = dlib.get_frontal_face_detector()
# ランドマーク検出器の初期化
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)

#EARの設定
def EAR(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear
#Eye Aspect Ratioが閾値を下回れば瞬きとしてカウントする
EAR_thresh=0.2
#瞬き状態が10秒続けば寝ていると判断する
EAR_consec_frames=cv2.CAP_PROP_FPS*15
EAR_thresh_counter=0
(lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

t=time.time()
while True:
    # フレームの読み込み
    ret, frame = cap.read()
    
    if ret:
        frame=imutils.resize(frame,width=450)
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔の検出
        faces = face_detector(gray)

        for face in faces:
            # 目の位置を検出
            landmarks = landmark_detector(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # 瞳孔の検出
            leftEye=landmarks[lstart:lend]
            rightEye=landmarks[rstart:rend]

            # まばたきの検出
            leftEAR=EAR(leftEye)
            rightEAR=EAR(rightEye)

            ear=(leftEAR+rightEAR)/2.0
            #瞬きのカウントが十分溜まれば寝ていることになる
            if ear<EAR_thresh:
                EAR_thresh_counter+=1
                if EAR_thresh_counter>=EAR_consec_frames:
                    #寝ているという画面上での注意喚起
                    #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "You're falling asleep!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), 2)
                    #cv2.putText(frame, "Press SPACE to snooze", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), 2)
                    #起こすための音声
                    while cv2.waitKey(1) & 0xFF != ord(' '):
                        cv2.imshow("Frame", frame)
                        playsound.playsound(r'C:\Users\rsato\Downloads\alarm.mp3')
                    EAR_thresh_counter=0
        
        # 結果の表示
        cv2.imshow("Frame", frame)
    if time.time()-t>=18:
        EAR_thresh_counter=0
        t=time.time()
    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
