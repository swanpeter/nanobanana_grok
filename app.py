import base64
import datetime
import io
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
try:
    from streamlit_cookies_controller import CookieController
except ImportError:
    CookieController = None

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except ImportError:
    StreamlitSecretNotFoundError = Exception

try:
    from google import genai
    from google.api_core import exceptions as google_exceptions
    from google.genai import types
    from google.cloud import storage
except ImportError:
    st.error(
        "必要なライブラリが不足しています。`pip install -r requirements.txt` を実行してください。"
    )
    st.stop()

def get_secret_value(key: str) -> Optional[str]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None
    except Exception:
        return None
    try:
        return secrets_obj[key]
    except (KeyError, TypeError, StreamlitSecretNotFoundError):
        pass
    get_method = getattr(secrets_obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key)
        except Exception:
            return None
    return None


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


TITLE = "Gemini 画像生成"
MODEL_NAME = "models/gemini-2.5-flash-image"
IMAGE_ASPECT_RATIO = "16:9"
COOKIE_KEY = "logged_in"
SESSION_COOKIE_KEY = "browser_session_id"
HISTORY_DIR = os.path.join(tempfile.gettempdir(), "nanobanana_history")
REFERENCE_IMAGES = [
    {
        "label": "春日俊彰さん",
        "path": os.path.join(os.path.dirname(__file__), "春日俊彰さん.jpeg"),
    },
    {
        "label": "公園おじさん",
        "path": os.path.join(os.path.dirname(__file__), "公園おじさん.jpg"),
    },
    {
        "label": "柴田理恵さん",
        "path": os.path.join(os.path.dirname(__file__), "柴田理恵さん.jpg"),
    },
    {
        "label": "鈴木雅之さん",
        "path": os.path.join(os.path.dirname(__file__), "鈴木雅之さん.png"),
    },
    {
        "label": "４コマ漫画",
        "path": os.path.join(os.path.dirname(__file__), "４コマ漫画.png"),
    },
]
DEFAULT_PROMPT_SUFFIX = (
    "((masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus))"
)
REFERENCE_EDIT_INSTRUCTION = (
    "referenceの男性をpromptのイメージに修正して、ただし構図は変えず、元画像に忠実にpromptの改変を加えてください"
)
NO_TEXT_TOGGLE_SUFFIX = (
    "((no background text, no symbols, no markings, no letters anywhere, no typography, "
    "no signboard, no watermark, no logo, no text, no subtitles, no labels, no poster elements, neutral background))"
)

DEFAULT_GEMINI_API_KEY = (
    get_secret_value("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
)


def _normalize_credential(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None




def get_secret_auth_credentials() -> Tuple[Optional[str], Optional[str]]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None, None
    except Exception:
        return None, None

    auth_section: Optional[Dict[str, Any]] = None
    if isinstance(secrets_obj, dict):
        auth_section = secrets_obj.get("auth")
    else:
        auth_section = getattr(secrets_obj, "get", lambda _key, _default=None: None)("auth")

    def _get_from_container(container: object, key: str) -> Optional[Any]:
        if isinstance(container, dict):
            return container.get(key)
        getter = getattr(container, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                try:
                    return getter(key, None)
                except TypeError:
                    return None
        try:
            return getattr(container, key)
        except AttributeError:
            return None

    def _extract_credential(container: object, keys: Tuple[str, ...]) -> Optional[Any]:
        for key in keys:
            value = _get_from_container(container, key)
            if value is not None:
                return value
        return None

    username = None
    password = None
    if auth_section is not None:
        username = _extract_credential(auth_section, ("username", "id", "user", "name"))
        password = _extract_credential(auth_section, ("password", "pass", "pwd"))

    if username is None:
        username = get_secret_value("USERNAME") or get_secret_value("ID")
    if password is None:
        password = get_secret_value("PASSWORD") or get_secret_value("PASS")

    normalized_username = _normalize_credential(str(username)) if username is not None else None
    normalized_password = _normalize_credential(str(password)) if password is not None else None
    return normalized_username, normalized_password


def get_configured_auth_credentials() -> Tuple[str, str]:
    secret_username, secret_password = get_secret_auth_credentials()
    if secret_username and secret_password:
        return secret_username, secret_password
    return "mezamashi", "mezamashi"


def _get_cookie_controller() -> Optional[object]:
    if CookieController is None:
        return None
    controller = st.session_state.get("_cookie_controller")
    if controller is None:
        try:
            controller = CookieController()
        except Exception:
            return None
        st.session_state["_cookie_controller"] = controller
    return controller


def sync_cookie_controller() -> None:
    controller = _get_cookie_controller()
    if controller is None:
        return
    sync_stage = st.session_state.get("_cookies_sync_stage", 0)
    if sync_stage == 0:
        try:
            controller.refresh()
        except Exception:
            return
        st.session_state["_cookies_sync_stage"] = 1
        rerun_app()
        return
    if sync_stage == 1:
        try:
            controller.refresh()
        except Exception:
            return
        st.session_state["_cookies_sync_stage"] = 2


def restore_login_from_cookie() -> bool:
    controller = _get_cookie_controller()
    if controller is None:
        return False
    for _ in range(2):
        try:
            controller.refresh()
            if controller.get(COOKIE_KEY) == "1":
                return True
        except Exception:
            return False
        time.sleep(0.3)
    return False


def persist_login_to_cookie(value: bool) -> None:
    controller = _get_cookie_controller()
    if controller is None:
        return
    try:
        if value:
            controller.set(COOKIE_KEY, "1")
            time.sleep(0.6)
        else:
            controller.remove(COOKIE_KEY)
    except Exception:
        return


def _get_history_path(session_id: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
    return os.path.join(HISTORY_DIR, f"{safe_id}.json")


def get_browser_session_id(create: bool = True) -> Optional[str]:
    controller = _get_cookie_controller()
    if controller is None:
        return None
    try:
        controller.refresh()
        session_id = controller.get(SESSION_COOKIE_KEY)
    except Exception:
        session_id = None
    if session_id:
        return str(session_id)
    if not create:
        return None
    new_id = uuid.uuid4().hex
    try:
        controller.set(SESSION_COOKIE_KEY, new_id)
        time.sleep(0.6)
    except Exception:
        return None
    return new_id


def _serialize_history(history: List[Dict[str, object]]) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for entry in history:
        image_bytes = entry.get("image_bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            image_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")
        else:
            image_b64 = None
        serialized.append(
            {
                "id": entry.get("id"),
                "prompt": entry.get("prompt"),
                "model": entry.get("model"),
                "no_text": entry.get("no_text"),
                "image_b64": image_b64,
            }
        )
    return serialized


def _deserialize_history(payload: List[Dict[str, object]]) -> List[Dict[str, object]]:
    history: List[Dict[str, object]] = []
    for entry in payload:
        image_b64 = entry.get("image_b64")
        image_bytes = decode_image_data(image_b64) if image_b64 else None
        history.append(
            {
                "id": entry.get("id"),
                "prompt": entry.get("prompt"),
                "model": entry.get("model"),
                "no_text": entry.get("no_text"),
                "image_bytes": image_bytes,
            }
        )
    return history


def load_history_from_storage() -> Optional[List[Dict[str, object]]]:
    session_id = get_browser_session_id(create=False)
    if not session_id:
        return None
    history_path = _get_history_path(session_id)
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    entries = payload.get("history")
    if not isinstance(entries, list):
        return None
    return _deserialize_history(entries)


def persist_history_to_storage() -> None:
    session_id = get_browser_session_id(create=True)
    if not session_id:
        return
    history_path = _get_history_path(session_id)
    payload = {
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "history": _serialize_history(st.session_state.history),
    }
    try:
        with open(history_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle)
    except Exception:
        return


def clear_history_storage() -> None:
    session_id = get_browser_session_id(create=False)
    if not session_id:
        return
    history_path = _get_history_path(session_id)
    try:
        if os.path.exists(history_path):
            os.remove(history_path)
    except Exception:
        return


def logout() -> None:
    st.session_state["authenticated"] = False
    persist_login_to_cookie(False)
    clear_history_storage()
    st.session_state.history = []
    rerun_app()


def inject_login_autofill_js() -> None:
    components.html(
        """
        <script>
        (function () {
            const parent = window.parent;
            if (!parent || !parent.document) {
                return;
            }
            const doc = parent.document;
            const inputs = Array.from(doc.querySelectorAll("input"));
            if (!inputs.length) {
                return;
            }
            let userInput = null;
            let passInput = null;
            for (const input of inputs) {
                const label = (input.getAttribute("aria-label") || "").toLowerCase();
                if (!userInput && (label === "id" || label === "user" || label === "username")) {
                    userInput = input;
                }
                if (!passInput && (label === "pass" || label === "password")) {
                    passInput = input;
                }
            }
            if (userInput) {
                userInput.setAttribute("name", "username");
                userInput.setAttribute("autocomplete", "username");
            }
            if (passInput) {
                passInput.setAttribute("name", "password");
                passInput.setAttribute("autocomplete", "current-password");
            }
            const form = userInput ? userInput.form : null;
            if (form) {
                form.setAttribute("autocomplete", "on");
            }
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def require_login() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"] and restore_login_from_cookie():
        st.session_state["authenticated"] = True
        get_browser_session_id(create=True)

    if st.session_state["authenticated"]:
        return

    st.title("ログイン")

    username, password = get_configured_auth_credentials()
    if not username or not password:
        st.info("ログイン情報が未設定です。管理者に連絡してください。")
        st.stop()
        return

    with st.form("login_form", clear_on_submit=False):
        input_username = st.text_input("ID")
        input_password = st.text_input("PASS", type="password")
        submitted = st.form_submit_button("ログイン")

    inject_login_autofill_js()

    if submitted:
        if input_username == username and input_password == password:
            st.session_state["authenticated"] = True
            persist_login_to_cookie(True)
            get_browser_session_id(create=True)
            st.success("ログインしました。")
            rerun_app()
            return
        st.error("IDまたはPASSが正しくありません。")
    st.stop()


def get_current_api_key() -> Optional[str]:
    api_key = st.session_state.get("config_api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return DEFAULT_GEMINI_API_KEY


def load_configured_api_key() -> str:
    return get_current_api_key() or ""


def decode_image_data(data: Optional[object]) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except (ValueError, TypeError):
            return None
    return None


def load_reference_image_bytes(path: str) -> Optional[bytes]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as file_handle:
            return file_handle.read()
    except Exception:
        return None


def get_image_dimensions(image_bytes: Optional[bytes]) -> Optional[Tuple[int, int]]:
    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.width, image.height
    except Exception:
        return None


def resize_image_bytes_to_height(image_bytes: Optional[bytes], target_height: int) -> Optional[bytes]:
    if not image_bytes:
        return None
    if target_height <= 0:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if height <= 0:
                return image_bytes
            scale = target_height / height
            target_width = max(1, int(width * scale))
            resized = image.resize((target_width, target_height), Image.LANCZOS)
            buffer = io.BytesIO()
            save_format = image.format if image.format else "PNG"
            resized.save(buffer, format=save_format)
            return buffer.getvalue()
    except Exception:
        return image_bytes


def extract_parts(candidate: object) -> Sequence:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if parts is None and isinstance(candidate, dict):
        parts = candidate.get("content", {}).get("parts", [])
    return parts or []


def collect_image_bytes(response: object) -> Optional[bytes]:
    visited: set[int] = set()
    queue: List[object] = []

    if response is not None:
        queue.append(response)

    def handle_inline(container: object) -> Optional[bytes]:
        if container is None:
            return None
        data = getattr(container, "data", None)
        if data is None and isinstance(container, dict):
            data = container.get("data")
        return decode_image_data(data)

    def maybe_file_data(container: object) -> Optional[bytes]:
        if container is None:
            return None
        file_data = getattr(container, "file_data", None)
        if file_data is None and isinstance(container, dict):
            file_data = container.get("file_data")
        if file_data:
            data = getattr(file_data, "data", None)
            if data is None and isinstance(file_data, dict):
                data = file_data.get("data")
            decoded = decode_image_data(data)
            if decoded:
                return decoded
        return None

    base64_charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")

    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        if isinstance(current, bytes):
            if current:
                return current
            continue

        if isinstance(current, (bytearray, memoryview)):
            as_bytes = bytes(current)
            if as_bytes:
                return as_bytes
            continue

        if isinstance(current, str):
            candidate = current.strip()
            if len(candidate) > 80 and set(candidate) <= base64_charset:
                decoded = decode_image_data(candidate)
                if decoded:
                    return decoded
            continue

        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if isinstance(current, dict):
            inline = current.get("inline_data")
            decoded = handle_inline(inline)
            if decoded:
                return decoded

            decoded = maybe_file_data(current)
            if decoded:
                return decoded

            for key, value in current.items():
                if key in {"data", "image", "blob"}:
                    decoded = decode_image_data(value)
                    if decoded:
                        return decoded
                queue.append(value)
            continue

        decoded = handle_inline(getattr(current, "inline_data", None))
        if decoded:
            return decoded

        decoded = maybe_file_data(current)
        if decoded:
            return decoded

        for attr in (
            "candidates",
            "content",
            "parts",
            "generated_content",
            "contents",
            "responses",
            "messages",
            "media",
            "image",
            "images",
        ):
            value = getattr(current, attr, None)
            if value is not None:
                queue.append(value)

        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray, memoryview)):
            queue.extend(list(current))

    return None


def collect_text_parts(response: object) -> List[str]:
    texts: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        for part in extract_parts(candidate):
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                texts.append(text)
    return texts


def _get_from_container(container: object, key: str) -> Optional[Any]:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except TypeError:
            try:
                return getter(key, None)
            except TypeError:
                return None
    try:
        return getattr(container, key)
    except AttributeError:
        return None


def sanitize_filename_component(value: str, max_length: int = 80) -> str:
    text = value or ""
    sanitized_chars: List[str] = []
    for char in text:
        if char in {"\n", "\r"}:
            sanitized_chars.append("-n-")
            continue
        if ord(char) < 32:
            continue
        if char in {'\\', '/', ':', '*', '?', '"', '<', '>', '|'}:
            continue
        if char.isspace():
            sanitized_chars.append("_")
            continue
        sanitized_chars.append(char)
    sanitized = "".join(sanitized_chars).strip("_")
    if not sanitized:
        sanitized = "prompt"
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def build_prompt_based_filename(prompt_text: str) -> str:
    prompt_component = sanitize_filename_component(prompt_text or "prompt", max_length=80)
    unique_suffix = uuid.uuid4().hex
    return f"user01_{prompt_component}_{unique_suffix}.png"


def upload_image_to_gcs(
    image_bytes: bytes,
    filename_prefix: str = "gemini_image",
    object_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if not image_bytes:
        return None, None

    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        st.warning("GCPの設定が見つからないためアップロードをスキップしました。")
        return None, None
    except Exception as exc:  # noqa: BLE001
        st.error(f"GCPの設定取得時にエラーが発生しました: {exc}")
        return None, None

    gcp_section = None
    if isinstance(secrets_obj, dict):
        gcp_section = secrets_obj.get("gcp")
    else:
        gcp_section = _get_from_container(secrets_obj, "gcp")

    if not gcp_section:
        st.warning("GCPの設定が見つからないためアップロードをスキップしました。")
        return None, None

    bucket_name = _get_from_container(gcp_section, "bucket_name")
    service_account_json = _get_from_container(gcp_section, "service_account_json")
    project_id = _get_from_container(gcp_section, "project_id")

    if not bucket_name or not service_account_json:
        st.warning("GCPの設定のうち bucket_name または service_account_json が不足しています。")
        return None, None

    service_account_info: Optional[Dict[str, Any]] = None
    if isinstance(service_account_json, (dict,)):
        service_account_info = dict(service_account_json)
    elif isinstance(service_account_json, (str, bytes)):
        raw_json = service_account_json.decode("utf-8") if isinstance(service_account_json, bytes) else service_account_json
        raw_json = raw_json.strip()
        try:
            service_account_info = json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                service_account_info = json.loads(raw_json, strict=False)
            except json.JSONDecodeError as exc:
                st.error(f"service_account_json の読み込みに失敗しました: {exc}")
                return None, None
    else:
        st.error("service_account_json の形式が不明です。文字列または辞書で設定してください。")
        return None, None

    if not isinstance(service_account_info, dict):
        st.error("service_account_json の内容が辞書形式ではありません。")
        return None, None

    try:
        storage_client = storage.Client.from_service_account_info(
            service_account_info,
            project=str(project_id) if project_id else None,
        )
        bucket = storage_client.bucket(str(bucket_name))
        if object_name:
            cleaned_object_name = object_name.strip()
            if not cleaned_object_name.lower().endswith(".png"):
                cleaned_object_name = f"{cleaned_object_name}.png"
            cleaned_object_name = cleaned_object_name.replace("/", "_").replace("\\", "_")
            filename = f"images/{cleaned_object_name}"
        else:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"images/{filename_prefix}_{timestamp}_{uuid.uuid4().hex}.png"
        blob = bucket.blob(filename)
        blob.upload_from_file(io.BytesIO(image_bytes), content_type="image/png")

        gcs_path = f"gs://{bucket.name}/{filename}"
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
        )
        return gcs_path, signed_url
    except Exception as exc:  # noqa: BLE001
        st.error(f"GCSへのアップロードに失敗しました: {exc}")
        return None, None


def init_history() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if not st.session_state.get("_history_loaded"):
        restored = load_history_from_storage()
        if restored is not None:
            st.session_state.history = restored
            st.session_state["_history_loaded"] = True
        else:
            if get_browser_session_id(create=False) is not None or _get_cookie_controller() is None:
                st.session_state["_history_loaded"] = True


def ensure_lightbox_assets() -> None:
    components.html(
        """
        <script>
        (function () {
            const parentWindow = window.parent;
            if (!parentWindow) {
                return;
            }

            try {
                delete parentWindow.__streamlitLightbox;
            } catch (err) {
                parentWindow.__streamlitLightbox = undefined;
            }
            parentWindow.__streamlitLightboxInitialized = false;
            const doc = parentWindow.document;

            if (!doc.getElementById("streamlit-lightbox-style")) {
                const style = doc.createElement("style");
                style.id = "streamlit-lightbox-style";
                style.textContent = `
                .streamlit-lightbox-thumb {
                    width: 100%;
                    display: block;
                    border-radius: 12px;
                    cursor: pointer;
                    transition: transform 0.16s ease-in-out;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
                    margin: 0 auto 0.75rem auto;
                }
                .streamlit-lightbox-thumb:hover {
                    transform: scale(1.02);
                }
                `;
                doc.head.appendChild(style);
            }

            parentWindow.__streamlitLightbox = (function () {
                let overlay = null;
                let keyHandler = null;

                function hide() {
                    if (!overlay) {
                        return;
                    }
                    overlay.style.opacity = "0";
                    const originalOverflow = overlay.getAttribute("data-original-overflow") || "";
                    doc.body.style.overflow = originalOverflow;
                    setTimeout(function () {
                        if (overlay && overlay.parentNode) {
                            overlay.parentNode.removeChild(overlay);
                        }
                        overlay = null;
                    }, 180);
                    if (keyHandler) {
                        parentWindow.removeEventListener("keydown", keyHandler);
                        keyHandler = null;
                    }
                }

                function show(src) {
                    hide();
                    overlay = doc.createElement("div");
                    overlay.id = "streamlit-lightbox-overlay";
                    overlay.style.position = "fixed";
                    overlay.style.zIndex = "10000";
                    overlay.style.top = "0";
                    overlay.style.left = "0";
                    overlay.style.right = "0";
                    overlay.style.bottom = "0";
                    overlay.style.display = "flex";
                    overlay.style.justifyContent = "center";
                    overlay.style.alignItems = "center";
                    overlay.style.background = "rgba(0, 0, 0, 0.92)";
                    overlay.style.cursor = "zoom-out";
                    overlay.style.opacity = "0";
                    overlay.style.transition = "opacity 0.18s ease-in-out";
                    overlay.setAttribute("data-original-overflow", doc.body.style.overflow || "");
                    doc.body.style.overflow = "hidden";

                    const full = doc.createElement("img");
                    full.src = src;
                    full.alt = "Generated image fullscreen";
                    full.style.maxWidth = "100vw";
                    full.style.maxHeight = "100vh";
                    full.style.objectFit = "contain";
                    full.style.boxShadow = "0 20px 45px rgba(0, 0, 0, 0.5)";
                    full.style.borderRadius = "0";

                    overlay.appendChild(full);
                    overlay.addEventListener("click", hide);

                    keyHandler = function (event) {
                        if (event.key === "Escape") {
                            hide();
                        }
                    };
                    parentWindow.addEventListener("keydown", keyHandler);

                    doc.body.appendChild(overlay);
                    requestAnimationFrame(function () {
                        overlay.style.opacity = "1";
                    });
                }

                return { show, hide };
            })();
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_clickable_image(image_bytes: bytes, element_id: str) -> None:
    ensure_lightbox_assets()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    image_src = f"data:image/png;base64,{encoded}"
    image_src_json = json.dumps(image_src)
    components.html(
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        img {{
            width: 100%;
            display: block;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.16s ease-in-out;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        }}
        img:hover {{
            transform: scale(1.02);
        }}
    </style>
</head>
<body>
    <img id="thumb" src="{image_src}" alt="Generated image">
    <script>
    (function() {{
        const img = document.getElementById("thumb");
        if (!img) {{
            return;
        }}

        function resizeFrame() {{
            const frame = window.frameElement;
            if (!frame) {{
                return;
            }}
            const frameWidth = frame.getBoundingClientRect().width || img.naturalWidth || img.clientWidth || 0;
            const ratio = img.naturalWidth ? (img.naturalHeight / Math.max(img.naturalWidth, 1)) : (img.clientHeight / Math.max(img.clientWidth, 1) || 1);
            const height = frameWidth ? Math.max(160, frameWidth * ratio) : (img.clientHeight || img.naturalHeight || 320);
            frame.style.height = height + "px";
        }}

        if (img.complete) {{
            resizeFrame();
        }} else {{
            img.addEventListener("load", resizeFrame);
        }}
        window.addEventListener("resize", resizeFrame);
        setTimeout(resizeFrame, 60);

        img.addEventListener("click", function() {{
            if (window.parent && window.parent.__streamlitLightbox) {{
                window.parent.__streamlitLightbox.show({image_src_json});
            }}
        }});
    }})();
    </script>
</body>
</html>
""",
        height=400,
        scrolling=False,
    )


def render_history() -> None:
    if not st.session_state.history:
        return

    st.subheader("履歴")
    for entry in st.session_state.history:
        image_bytes = entry.get("image_bytes")
        prompt_text = entry.get("prompt") or ""
        if image_bytes:
            image_id = entry.get("id")
            if not isinstance(image_id, str):
                image_id = f"img_{uuid.uuid4().hex}"
                entry["id"] = image_id
            render_clickable_image(image_bytes, image_id)
        prompt_display = prompt_text.strip()
        st.markdown("**Prompt**")
        if prompt_display:
            st.text(prompt_display)
        else:
            st.text("(未入力)")
        st.divider()


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🧠", layout="centered")
    sync_cookie_controller()
    init_history()
    require_login()

    with st.sidebar:
        if st.button("ログアウト"):
            logout()

    st.title("脳内大喜利")

    api_key = load_configured_api_key()

    prompt = st.text_area("Prompt", height=150, placeholder="描いてほしい内容を入力してください")
    reference_index = st.radio(
        "参照画像を選択",
        options=list(range(len(REFERENCE_IMAGES))),
        format_func=lambda idx: REFERENCE_IMAGES[idx]["label"],
        horizontal=True,
    )
    try:
        reference_index_value = int(reference_index)
    except (TypeError, ValueError):
        reference_index_value = -1
    reference_entry = (
        REFERENCE_IMAGES[reference_index_value]
        if 0 <= reference_index_value < len(REFERENCE_IMAGES)
        else None
    )
    reference_path = reference_entry["path"] if reference_entry else None
    reference_thumb = load_reference_image_bytes(reference_path) if reference_path else None
    if reference_thumb:
        resized_thumb = resize_image_bytes_to_height(reference_thumb, 200)
        if reference_path:
            caption = os.path.splitext(os.path.basename(reference_path))[0]
        elif reference_entry:
            caption = str(reference_entry.get("label", ""))
        else:
            caption = ""
        st.image(resized_thumb, caption=caption)
    else:
        st.warning("参照画像のサムネイルを読み込めませんでした。")

    if st.button("Generate", type="primary"):
        if not api_key:
            st.warning("Gemini API key が設定されていません。Streamlit secrets などで設定してください。")
            st.stop()
        if not prompt.strip():
            st.warning("プロンプトを入力してください。")
            st.stop()
        if not reference_path:
            st.error("参照画像が未選択です。")
            st.stop()

        client = genai.Client(api_key=api_key.strip())
        stripped_prompt = prompt.rstrip()
        prompt_components: List[str] = []
        if stripped_prompt:
            prompt_components.append(stripped_prompt)
        prompt_components.append(REFERENCE_EDIT_INSTRUCTION)
        prompt_components.extend([DEFAULT_PROMPT_SUFFIX, NO_TEXT_TOGGLE_SUFFIX])
        prompt_for_request = "\n".join(prompt_components)
        reference_image_bytes = load_reference_image_bytes(reference_path)
        if not reference_image_bytes:
            st.error("参照画像を読み込めませんでした。")
            st.stop()

        with st.spinner("画像を生成しています..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        prompt_for_request,
                        types.Part.from_bytes(data=reference_image_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=IMAGE_ASPECT_RATIO),
                    ),
                )
            except google_exceptions.ResourceExhausted:
                st.error(
                    "Gemini API のクォータ（無料枠または請求プラン）を超えました。"
                    "しばらく待つか、Google AI Studio で利用状況と請求設定を確認してください。"
                )
                st.info("https://ai.google.dev/gemini-api/docs/rate-limits")
                st.stop()
            except google_exceptions.GoogleAPICallError as exc:
                st.error(f"API 呼び出しに失敗しました: {exc.message}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"予期しないエラーが発生しました: {exc}")
                st.stop()

        image_bytes = collect_image_bytes(response)
        if not image_bytes:
            st.error("画像データを取得できませんでした。")
            st.stop()

        user_prompt = prompt.strip()
        object_name = build_prompt_based_filename(user_prompt)
        upload_image_to_gcs(image_bytes, object_name=object_name)

        st.session_state.history.insert(
            0,
            {
                "id": f"img_{uuid.uuid4().hex}",
                "image_bytes": image_bytes,
                "prompt": user_prompt,
                "model": MODEL_NAME,
                "no_text": True,
            },
        )
        persist_history_to_storage()
        st.success("生成完了")

    render_history()


if __name__ == "__main__":
    main()
