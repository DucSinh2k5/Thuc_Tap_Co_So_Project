import io
from pathlib import Path

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR
DEFAULT_MODEL_NAME = "ResNet50.pkl"

DEFAULT_CLASS_NAMES = ["minor", "moderate", "severe"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_INPUT_SIZES = {
	"Efficient_B2.pkl": 260,
}


def list_model_files() -> list[Path]:
	return sorted(MODEL_DIR.glob("*.pkl"))


def infer_class_names() -> list[str]:
	training_dir = BASE_DIR / "training"
	if not training_dir.exists():
		return DEFAULT_CLASS_NAMES

	class_dirs = sorted([p for p in training_dir.iterdir() if p.is_dir()])
	names: list[str] = []
	for folder in class_dirs:
		name = folder.name.strip().lower()
		if "-" in name:
			name = name.split("-", 1)[1].strip()
		names.append(name)

	if len(names) == 3:
		return names
	return DEFAULT_CLASS_NAMES


def build_transform(model_name: str) -> transforms.Compose:
	img_size = MODEL_INPUT_SIZES.get(model_name, 224)
	return transforms.Compose(
		[
			transforms.Lambda(lambda img: img.convert("RGB")),
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		]
	)


def _extract_state_dict(state):
	if isinstance(state, dict):
		for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
			if key in state and isinstance(state[key], dict):
				return state[key]
		return state
	return None


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
	if all(key.startswith(prefix) for key in state_dict.keys()):
		return {key[len(prefix):]: value for key, value in state_dict.items()}
	return state_dict


def _count_stage_blocks(state_dict: dict, stage_idx: int) -> int:
	prefix = f"features.{stage_idx}."
	indices: set[int] = set()
	for key in state_dict.keys():
		if not key.startswith(prefix):
			continue
		rest = key[len(prefix):]
		first = rest.split(".", 1)[0]
		if first.isdigit():
			indices.add(int(first))
	if not indices:
		return 0
	return max(indices) + 1


def _infer_convnext_variant(state_dict: dict | None) -> str:
	if not isinstance(state_dict, dict):
		return "convnext_tiny"
	stem_weight = state_dict.get("features.0.0.weight")
	if isinstance(stem_weight, torch.Tensor) and stem_weight.ndim >= 1:
		out_channels = int(stem_weight.shape[0])
		if out_channels == 128:
			return "convnext_base"
		if out_channels == 192:
			return "convnext_large"
		if out_channels == 96:
			stage3_blocks = _count_stage_blocks(state_dict, 5)
			return "convnext_small" if stage3_blocks >= 20 else "convnext_tiny"
	return "convnext_tiny"


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
	if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
		for idx in range(len(model.classifier) - 1, -1, -1):
			layer = model.classifier[idx]
			if isinstance(layer, nn.Linear):
				model.classifier[idx] = nn.Linear(layer.in_features, num_classes)
				return
	if hasattr(model, "fc") and isinstance(model.fc, nn.Sequential):
		for idx in range(len(model.fc) - 1, -1, -1):
			layer = model.fc[idx]
			if isinstance(layer, nn.Linear):
				model.fc[idx] = nn.Linear(layer.in_features, num_classes)
				return
	if hasattr(model, "head") and isinstance(model.head, nn.Linear):
		model.head = nn.Linear(model.head.in_features, num_classes)
		return
	if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return
	if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
		model.classifier = nn.Linear(model.classifier.in_features, num_classes)
		return
	raise RuntimeError("Khong tim thay lop phan loai de thay the")


def _build_convnext(variant: str, num_classes: int) -> nn.Module:
	builders = {
		"convnext_tiny": models.convnext_tiny,
		"convnext_small": models.convnext_small,
		"convnext_base": models.convnext_base,
		"convnext_large": models.convnext_large,
	}
	if variant not in builders:
		variant = "convnext_tiny"
	model = builders[variant](weights=None)
	_replace_classifier(model, num_classes)
	return model


def _build_efficientnet(variant: str, num_classes: int) -> nn.Module:
	builders = {
		"efficientnet_b0": models.efficientnet_b0,
		"efficientnet_b2": models.efficientnet_b2,
	}
	if variant not in builders:
		variant = "efficientnet_b0"
	model = builders[variant](weights=None)
	_replace_classifier(model, num_classes)
	return model


def _build_resnet(variant: str, num_classes: int, fc_sequential: bool) -> nn.Module:
	builders = {
		"resnet18": models.resnet18,
		"resnet50": models.resnet50,
	}
	if variant not in builders:
		variant = "resnet50"
	model = builders[variant](weights=None)
	if fc_sequential:
		in_features = model.fc.in_features
		model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
		return model
	_replace_classifier(model, num_classes)
	return model


def _build_model_for_name(model_name: str, state_dict: dict, num_classes: int) -> nn.Module:
	if model_name == "cnn_car.pkl":
		fc_sequential = "fc.1.weight" in state_dict or "fc.1.bias" in state_dict
		model = _build_resnet("resnet18", num_classes, fc_sequential)
	elif model_name == "ResNet50.pkl":
		fc_sequential = "fc.1.weight" in state_dict or "fc.1.bias" in state_dict
		model = _build_resnet("resnet50", num_classes, fc_sequential)
	elif model_name == "Efficient_B0.pkl":
		model = _build_efficientnet("efficientnet_b0", num_classes)
	elif model_name == "Efficient_B2.pkl":
		model = _build_efficientnet("efficientnet_b2", num_classes)
	elif model_name == "ConvNeXt.pkl":
		variant = _infer_convnext_variant(state_dict)
		model = _build_convnext(variant, num_classes)
	else:
		raise RuntimeError(f"Chua ho tro model: {model_name}")

	model.load_state_dict(state_dict, strict=True)
	return model


@st.cache_resource
def load_model(model_path: str, num_classes: int):
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Khong tim thay model: {path}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state = torch.load(path, map_location=device)

	if isinstance(state, nn.Module):
		model = state
		try:
			_replace_classifier(model, num_classes)
		except RuntimeError:
			pass
	else:
		state_dict = _extract_state_dict(state)
		if not isinstance(state_dict, dict):
			raise RuntimeError("Checkpoint khong chua state_dict hop le")
		state_dict = _strip_prefix(state_dict, "module.")
		state_dict = _strip_prefix(state_dict, "model.")
		state_dict = _strip_prefix(state_dict, "net.")
		model = _build_model_for_name(path.name, state_dict, num_classes)

	model.to(device)
	model.eval()
	return model, device


def fetch_image_from_url(url: str) -> Image.Image:
	headers = {"User-Agent": "Mozilla/5.0"}
	response = requests.get(url, timeout=20, headers=headers)
	response.raise_for_status()
	return Image.open(io.BytesIO(response.content)).convert("RGB")


def load_uploaded_image(uploaded_file) -> Image.Image:
	if uploaded_file is None:
		raise ValueError("Chua chon anh upload")
	return Image.open(uploaded_file).convert("RGB")


def predict_image(image: Image.Image, model: nn.Module, device: torch.device, transform):
	tensor = transform(image).unsqueeze(0).to(device)

	with torch.no_grad():
		logits = model(tensor)
		probs = torch.softmax(logits, dim=1)[0].cpu()

	pred_idx = int(torch.argmax(probs).item())
	return pred_idx, probs


def main() -> None:
	st.set_page_config(page_title="Car Damage Severity", page_icon="car", layout="centered")
	st.title("Phan loai muc do hu hong xe")
	st.write("Nhap URL anh hoac upload anh tu may, he thong se du doan 1 trong 3 nhom: minor, moderate, severe.")

	model_files = list_model_files()
	if not model_files:
		st.error(f"Khong tim thay file .pkl trong: {MODEL_DIR}")
		st.stop()

	model_names = [path.name for path in model_files]
	default_index = model_names.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in model_names else 0

	st.sidebar.header("Model")
	selected_name = st.sidebar.selectbox("Chon model", model_names, index=default_index)
	model_path = MODEL_DIR / selected_name
	if st.sidebar.button("Reload model"):
		load_model.clear()

	input_size = MODEL_INPUT_SIZES.get(selected_name, 224)
	st.sidebar.caption(f"Model: {model_path.name}")
	st.sidebar.caption(f"Path: {model_path}")
	st.sidebar.caption(f"Input size: {input_size}x{input_size}")

	class_names = infer_class_names()
	st.caption(f"Model: {model_path}")

	try:
		model, device = load_model(str(model_path), len(class_names))
	except Exception as exc:
		st.error(f"Khong load duoc model: {exc}")
		st.stop()

	input_mode = st.radio("Nguon anh", options=["URL", "Upload tu may"], horizontal=True)
	image_url = ""
	uploaded_file = None

	if input_mode == "URL":
		image_url = st.text_input("URL anh", placeholder="https://example.com/car.jpg")
	else:
		uploaded_file = st.file_uploader(
			"Chon anh tu may",
			type=["jpg", "jpeg", "png", "bmp", "webp"],
		)
		if uploaded_file is not None:
			st.caption(f"Da chon file: {uploaded_file.name}")

	if st.button("Phan loai", type="primary"):
		try:
			with st.spinner("Dang xu ly anh va du doan..."):
				if input_mode == "URL":
					if not image_url.strip():
						st.warning("Vui long nhap URL anh hop le.")
						return
					image = fetch_image_from_url(image_url.strip())
				else:
					if uploaded_file is None:
						st.warning("Vui long chon anh tu may.")
						return
					image = load_uploaded_image(uploaded_file)

				transform = build_transform(selected_name)
				pred_idx, probs = predict_image(image, model, device, transform)

			st.image(image, caption="Anh dau vao", use_container_width=True)

			label = class_names[pred_idx]
			confidence = float(probs[pred_idx].item()) * 100.0

			st.success(f"Ket qua: {label.upper()} ({confidence:.2f}%)")
			st.subheader("Xac suat theo tung nhom")
			prob_dict = {
				class_names[i]: round(float(probs[i].item()) * 100.0, 2)
				for i in range(len(class_names))
			}
			st.write(prob_dict)
			st.progress(min(max(confidence / 100.0, 0.0), 1.0))

		except FileNotFoundError as exc:
			st.error(str(exc))
		except requests.RequestException as exc:
			st.error(f"Khong tai duoc anh tu URL: {exc}")
		except UnidentifiedImageError:
			if input_mode == "URL":
				st.error("Noi dung URL khong phai file anh hop le.")
			else:
				st.error("File upload khong phai anh hop le.")
		except ValueError as exc:
			st.error(str(exc))
		except RuntimeError as exc:
			st.error(f"Loi khi nap model/state_dict: {exc}")
		except Exception as exc:
			st.error(f"Co loi khi du doan: {exc}")


if __name__ == "__main__":
	main()