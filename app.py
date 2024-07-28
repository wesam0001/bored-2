import base64
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from cvzone.PoseModule import PoseDetector
import cv2

app = FastAPI()
templates = Jinja2Templates(directory=".")

detector = PoseDetector(staticMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=True, smoothSegmentation=True, detectionCon=0.7, trackCon=0.4)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_frame")
async def process_frame(request: Request):
    data = await request.json()
    image_data = data['image'].split(',')[1]
    image = base64.b64decode(image_data)
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
    
    if lmList:
        RBack_angle, img = detector.findAngle(lmList[11][0:2], lmList[23][0:2], lmList[25][0:2], img=img, color=(0, 0, 255), scale=1)
        RKnee_angle, img = detector.findAngle(lmList[23][0:2], lmList[25][0:2], lmList[27][0:2], img=img, color=(0, 0, 255), scale=1)
        
        SP = "Starting Position"
        BP = "Correct Back Position!!"
        KP = "Correct Knee Position!!"
        IBP = "Incorrect Back Position!!"
        IKP = "Incorrect Knee Position!!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        RP = (0, 255, 0)
        WP = (0, 0, 255)
        thickness = 3
        
        if (RBack_angle > 170 and RBack_angle <= 185):
            cv2.putText(img, SP, (50, 50), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
        elif (RBack_angle > 295 or RBack_angle < 280):
            cv2.putText(img, IBP, (50, 100), font, fontScale, WP, thickness, cv2.LINE_AA)
            if (RKnee_angle > 80 or RKnee_angle < 70):
                cv2.putText(img, IKP, (50, 150), font, fontScale, WP, thickness, cv2.LINE_AA)
        elif (RBack_angle > 280 and RBack_angle <= 295):
            cv2.putText(img, BP, (50, 100), font, fontScale, RP, thickness, cv2.LINE_AA)
            if (RKnee_angle < 80 and RKnee_angle >= 70):
                cv2.putText(img, KP, (50, 150), font, fontScale, RP, thickness, cv2.LINE_AA)
    
    ret, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return JSONResponse({'image': img_str})
