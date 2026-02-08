import streamlit as st
import pytesseract
import cv2
import pandas as pd
import numpy as np
import sqlite3
import json
from PIL import Image
import io
import re
from datetime import datetime
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

DB_PATH = "data/sample_receipts/history.db"
# ---------- DATABASE SETUP ----------

def init_db():
    os.makedirs("data/sample_receipts", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            total REAL,
            date TEXT,
            split_summary TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(filename, total, split_summary):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO receipts (filename, total, date, split_summary)
        VALUES (?, ?, ?, ?)
    """, (
        filename,
        total,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        split_summary
    ))
    conn.commit()
    conn.close()

def get_history():
    init_db()   
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM receipts", conn)
    conn.close()
    return df

def delete_bill(bill_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM receipts WHERE id = ?", (bill_id,))
    conn.commit()
    conn.close()

# ---------- OCR PROCESSING ----------
def extract_text(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def parse_items(text):
    lines = text.split("\n")
    items = []
    pattern = re.compile(r"([A-Za-z ]+)\s+(\d+\.\d{2})")
    for line in lines:
        match = pattern.search(line)
        if match:
            item, price = match.groups()
            items.append((item.strip(), float(price)))
    return items

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Smart Expense Splitter", page_icon="💸")
st.title("Smart Expense Splitter 💸")
st.write("Upload a bill and split expenses among friends!")

tab1, tab2 = st.tabs(["📤 Upload Bill", "  🕒 History"])

with tab1:
    uploaded_file = st.file_uploader("Upload a bill/receipt image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        from PIL import Image
        import cv2 
        import numpy as np
        import pytesseract
        import re
        image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image correctly
        st.image(image, caption="Uploaded Bill", use_container_width=True)

    # --- Preprocess for OCR ---
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Remove noise
        gray = cv2.medianBlur(gray, 3)

    # Dilate text slightly to connect broken digits
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)

    # --- OCR text extraction ---
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)

    # --- Show extracted text for debugging ---
        st.subheader("📄 Extracted OCR Text")
        st.text_area("Extracted Text", text, height=200)
    # --- Clean common OCR mistakes ---
        text = text.replace('S0', '50').replace('s0', '50')
        text = text.replace('80.', '50.').replace('B0', '80')
        text = re.sub(r'[^A-Za-z0-9\.\s:]', '', text)

    # --- Parse item names and prices using regex ---
        items = re.findall(r'([A-Za-z ]+)\s+\d+\s+(\d+\.\d{1,2})', text)

        if items:
            import pandas as pd
            df = pd.DataFrame(items, columns=["Item", "Price"])
            df["Price"] = df["Price"].astype(float)
            st.dataframe(df)

            st.subheader("💰 Expense Split Options")
            friends = st.text_input("Enter names separated by commas")
            friends_list = [f.strip() for f in friends.split(",") if f.strip()]

            if st.button("Split Equally"):
                split_amount = round(df["Price"].sum() / len(friends_list), 2)
                result = {f: split_amount for f in friends_list}
                st.success(f"Each person pays ₹{split_amount}")
                st.json(result)
                save_to_db(uploaded_file.name, df["Price"].sum(), json.dumps(result))
        else:
            st.warning("No items found. Try a clearer image.")
    with tab2:
        st.subheader("🧾 Previous Receipts")

        history = get_history()

        if not history.empty:
            for _, row in history.iterrows():

                st.markdown("### 🧾 Bill")

                st.write("📄 File:", row["filename"])
                st.write("📅 Date:", row["date"])
                st.write("💰 Total:", row["total"])

                st.markdown("**Split Details:**")

                try:
                    split_data = json.loads(row["split_summary"])
                except json.JSONDecodeError:
                    split_data = eval(row["split_summary"])

                split_df = pd.DataFrame(
                    list(split_data.items()),
                    columns=["Name", "Amount to Pay"]
                )

                st.dataframe(split_df)

                # 🔥 DELETE BUTTON
                if st.button("🗑 Delete this bill", key=f"delete_{row['id']}"):
                    delete_bill(row["id"])
                    st.success("Bill deleted!")
                    import time
                    time.sleep(1)
                    st.rerun()
                st.divider()
        else:
            st.info("No receipts saved yet.")

