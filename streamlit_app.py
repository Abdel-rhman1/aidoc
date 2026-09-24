from __future__ import annotations

import requests
import streamlit as st


MAX_UPLOAD_BYTES = 8 * 1024 * 1024

st.set_page_config(page_title="Egyptian ID OCR", layout="centered")
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000/verify")
st.title("Egyptian National ID OCR")

uploaded = st.file_uploader("Upload national ID front image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("Image is too large. Please upload an image smaller than 8 MB.")
        st.stop()

    st.image(uploaded, use_container_width=True)

    if st.button("Verify ID", type="primary"):
        with st.spinner("Reading document..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            try:
                response = requests.post(API_URL, files=files, timeout=600)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
            except ValueError:
                st.error("API returned an invalid response.")
            else:
                if data["valid"]:
                    st.success("Valid Egyptian national ID structure")
                else:
                    st.warning("Could not verify a complete valid ID")

                col1, col2 = st.columns(2)
                col1.metric("Name", data.get("name") or "Not found")
                col2.metric("National ID", data.get("national_id") or "Not found")

                validation = data.get("validation", {})
                st.json(
                    {
                        "birth_date": validation.get("birth_date"),
                        "governorate": validation.get("governorate"),
                        "gender": validation.get("gender"),
                        "errors": validation.get("errors", []),
                    }
                )

                with st.expander("OCR text"):
                    st.text(data.get("ocr_text") or "")

