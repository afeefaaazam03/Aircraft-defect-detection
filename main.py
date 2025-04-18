from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import onnxruntime as ort
import cv2
import os
import uuid

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

IMG_SIZE = 640
YOLO_CLASSES = ['Corrosion', 'corrsion-surface-wearout', 'crack', 'crack-dent', 'dent', 'dent-corrosion', 'dent-surface wear', 'good', 'scratch', 'surface wear', 'wire fault', 'wire fault-dent']

session = ort.InferenceSession("C:\\Users\\HOMELP\\Desktop\\API\\best.onnx")

def preprocess(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    img_padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    img_padded[:nh, :nw] = img_resized
    img_input = img_padded.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, scale, (h, w), (nh, nw), img, nh, nw

def postprocess(output, scale, orig_shape, resized_shape, conf_thres=0.25, iou_thres=0.45):
    boxes = output[0][:, :4]
    scores = output[0][:, 4]
    class_probs = output[0][:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = scores * class_probs.max(axis=1)

    mask = confidences > conf_thres
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    boxes_xy = np.zeros_like(boxes)
    boxes_xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    gain = scale
    boxes_xy /= gain
    boxes_xy[:, [0, 2]] = np.clip(boxes_xy[:, [0, 2]], 0, orig_shape[1])
    boxes_xy[:, [1, 3]] = np.clip(boxes_xy[:, [1, 3]], 0, orig_shape[0])

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xy.tolist(),
        scores=confidences.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    results = []
    for i in indices:
        results.append({
            "box": boxes_xy[i].tolist(),
            "score": float(confidences[i]),
            "class_id": int(class_ids[i])
        })
    return results

def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        class_id = det["class_id"]
        label = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else str(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"{label} {det['score']:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_input, scale, orig_shape, resized_shape, orig_img, nh, nw = preprocess(image_bytes)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    detections = postprocess(outputs, scale, orig_shape, resized_shape)
    # Draw boxes
    img_annotated = draw_boxes(orig_img, detections)
    # Save annotated image to static folder
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("static", filename)
    cv2.imwrite(filepath, img_annotated)
    # Prepare detection info for the template
    for d in detections:
        d["label"] = YOLO_CLASSES[d["class_id"]] if d["class_id"] < len(YOLO_CLASSES) else str(d["class_id"])
        d["box"] = [int(x) for x in d["box"]]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {
                "image_url": f"/static/{filename}",
                "detections": detections,
                "filename": file.filename
            }
        }
    )