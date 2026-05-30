import io
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights" 
DEFAULT_MODEL_NAME = "best_merge_data_1061.pt"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Model file not found: {path}")
	return YOLO(str(path))


def read_image_from_url(url: str) -> Image.Image:
	response = requests.get(url, timeout=15)
	response.raise_for_status()
	return Image.open(io.BytesIO(response.content)).convert("RGB")


def run_inference(model: YOLO, image: Image.Image, conf: float, iou: float):
	results = model.predict(source=np.array(image), conf=conf, iou=iou, verbose=False)
	annotated_image = results[0].plot(pil=True)
	return results[0], annotated_image


def format_boxes(result) -> list[dict]:
	if result.boxes is None or len(result.boxes) == 0:
		return []

	rows = []
	for box in result.boxes:
		cls_id = int(box.cls.item())
		x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
		rows.append(
			{
				"class": result.names.get(cls_id, str(cls_id)),
				"conf": round(float(box.conf.item()), 4),
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
			}
		)
	return rows


def main():
	st.set_page_config(page_title="Damage Detection", layout="wide")
	st.title("Damage Detection - YOLO")
	st.write("Nhap URL anh hoac upload anh, sau do detect bounding box damage.")

	if "input_image" not in st.session_state:
		st.session_state.input_image = None
	if "image_name" not in st.session_state:
		st.session_state.image_name = ""

	st.sidebar.header("Model")
	if not WEIGHTS_DIR.exists():
		st.error(f"Khong tim thay thu muc weights: {WEIGHTS_DIR}")
		st.stop()

	model_files = sorted(WEIGHTS_DIR.glob("*.pt"))
	if not model_files:
		st.error(f"Khong tim thay file .pt trong: {WEIGHTS_DIR}")
		st.stop()

	model_names = [path.name for path in model_files]
	default_index = model_names.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in model_names else 0
	selected_name = st.sidebar.selectbox("Chon model", model_names, index=default_index)
	model_path = WEIGHTS_DIR / selected_name
	if st.sidebar.button("Reload model"):
		load_model.clear()

	st.sidebar.caption(f"Model: {model_path.name}")
	st.sidebar.caption(f"Path: {model_path}")

	try:
		model = load_model(str(model_path))
	except Exception as exc:
		st.error(f"Khong load duoc model: {exc}")
		st.stop()

	if getattr(model, "task", None) not in (None, "detect"):
		st.warning(f"Model task hien tai: {getattr(model, 'task', 'unknown')}. Box co the khong hien.")

	source = st.radio("Nguon anh", ["URL", "Upload"], horizontal=True)

	if source == "URL":
		image_url = st.text_input("Nhap URL anh")
		if st.button("Tai anh tu URL"):
			if not image_url.strip():
				st.warning("Vui long nhap URL hop le.")
			else:
				try:
					st.session_state.input_image = read_image_from_url(image_url.strip())
					st.session_state.image_name = image_url.strip()
				except (requests.RequestException, UnidentifiedImageError) as exc:
					st.error(f"Khong doc duoc anh tu URL: {exc}")

	if source == "Upload":
		uploaded_file = st.file_uploader(
			"Chon anh tu may", type=["jpg", "jpeg", "png", "bmp", "webp"]
		)
		if uploaded_file is not None:
			try:
				st.session_state.input_image = Image.open(uploaded_file).convert("RGB")
				st.session_state.image_name = uploaded_file.name
			except UnidentifiedImageError as exc:
				st.error(f"File anh khong hop le: {exc}")

	image = st.session_state.input_image
	if image is None:
		st.info("Chua co anh. Hay nhap URL hoac upload anh.")
		st.stop()

	st.subheader(f"Anh dau vao: {st.session_state.image_name}")
	conf = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
	iou = st.slider("IoU threshold", 0.05, 1.0, 0.45, 0.05)

	if st.button("Detect damage", type="primary"):
		with st.spinner("Dang detect..."):
			result, plotted = run_inference(model, image, conf, iou)

		col1, col2 = st.columns(2)
		with col1:
			st.image(image, caption="Anh goc", use_container_width=True)
		with col2:
			st.image(plotted, caption="Anh co bounding box", use_container_width=True)

		box_rows = format_boxes(result)
		st.success(f"Phat hien {len(box_rows)} bounding box damage.")
		if box_rows:
			st.dataframe(box_rows, use_container_width=True)


if __name__ == "__main__":
	main()
