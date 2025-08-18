import streamlit as st
import pandas as pd
import json
from io import BytesIO
import copy

# ---- Custom CSS for polish ----
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stDataFrame, .stDataEditor {background: #f8fafc; border-radius: 8px;}
    .stButton>button {background: #2563eb; color: white; font-weight: 600; border-radius: 6px;}
    .stDownloadButton>button {background: #059669; color: white; font-weight: 600; border-radius: 6px;}
    .stTextInput>div>div>input {border-radius: 6px;}
    .stSelectbox>div>div>div {border-radius: 6px;}
    .stExpanderHeader {font-size: 1.1rem; font-weight: 600;}
    .stSubheader {margin-top: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar ----
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=180)
st.sidebar.markdown("""
### 📝 Human Feedback Table Editor
- **Upload your JSON file
- **Select a category
- **Edit headers, plan and table cells
- **Download the updated JSON
""")
st.sidebar.info("All changes are local until you download the file.")

# ---- Page Setup ----
st.set_page_config(page_title="Human Feedback Editor", layout="wide")
st.title("🧑‍💻 Human Review: Table Parsing")
st.markdown(
    """
    <div style='color: #64748b; font-size: 1.1rem;'>
    Easily Upload Multiple JSON and Edit the Files.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Upload Multiple JSONs ----
st.markdown("---")
with st.container():
    st.subheader("\U0001F4E4Upload JSON files")
    uploaded_files = st.file_uploader(
        "Upload one or more JSON files", 
        type=["json"], 
        key="json_uploader", 
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload at least one JSON file to begin.")
        st.stop()

# ---- Display all uploaded JSONs with filenames ----
file_data = []
for uploaded_file in uploaded_files:
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
        file_data.append({"name": uploaded_file.name, "data": data, "file": uploaded_file})
    except Exception as e:
        st.error(f"Failed to parse {uploaded_file.name}: {e}")

# ---- Select which file to edit ----
st.markdown("---")
st.subheader("✏️ Select a file to edit")
file_names = [f["name"] for f in file_data]
selected_file_idx = st.selectbox("Choose a file to edit", options=range(len(file_names)), format_func=lambda i: file_names[i])
selected_file = file_data[selected_file_idx]
data = selected_file["data"]
uploaded_file = selected_file["file"]

# ---- Load and Parse JSON ----
# (No need to re-parse, already done above)
st.success(f"JSON loaded successfully from {uploaded_file.name}.")

# ---- Raw JSON view ----
with st.expander("\U0001F50D View Raw JSON", expanded=False):
    st.json(data)

# ---- Detect where the tables/categories live ----
table_block = data.get("table", {})
containers = []
container_map = []  # Track (type, index) for each container
container_key = None

# Always include the top-level table if it has rows/columns
if isinstance(table_block, dict) and (table_block.get("rows") or table_block.get("columns") or table_block.get("column_header")):
    containers.append(copy.deepcopy(table_block))
    container_map.append(("table", None))
    container_key = "single"
    # If the table has categories, add them as well
    if table_block.get("categories") and isinstance(table_block["categories"], list):
        for idx, cat in enumerate(table_block["categories"]):
            # Avoid adding the main table again if it's present in categories
            if cat is table_block or cat == table_block:
                continue
            containers.append(copy.deepcopy(cat))
            container_map.append(("category", idx))
        container_key = "categories"

# Add sub-tables if present
if isinstance(table_block, dict) and table_block.get("Tables"):
    containers.extend(table_block.get("Tables"))
    container_key = "Tables"

if not containers:
    # Fallback: check if the root data itself is a table
    if isinstance(data, dict) and data.get("columns") and data.get("rows"):
        containers.append(copy.deepcopy(data))
        container_map.append(("root", None))
        container_key = "root"
    else:
        st.error("No recognizable table structure found under data['table'] or at the root level.")
        st.stop()

# ---- Allow selecting which container to edit ----
st.markdown("---")
st.subheader("\U0001F4C1 Select Category")
container_labels = []
for idx, c in enumerate(containers):
    if idx == 0 and (c is table_block):
        label = "Top Level Table"
        pid = c.get("plan_id") or c.get("plan_name") or ""
        th = c.get("table_header") or c.get("row_header") or ""
        if pid or th:
            label += f" (plan_id={pid} | header={th})"
        container_labels.append(label)
    else:
        pid = c.get("plan_id") or c.get("plan_name") or "N/A"
        th = c.get("table_header") or c.get("row_header") or ""
        container_labels.append(f"{idx+1}: plan_id={pid} | header={th}")

selected_index = st.selectbox(
    "Select Category",
    options=list(range(len(containers))),
    format_func=lambda i: container_labels[i],
    index=0,
    key="container_selector",
    help="Choose which table/category to edit."
)

table_data = containers[selected_index]

# ---- Editable metadata ----
st.markdown("---")
st.subheader("📝 Headers")
col1, col2 = st.columns([1,1])
with col1:
    table_plan_id = table_data.get("plan_id") or ""
    new_table_plan_id = st.text_input("Table Plan Name", 
                                    value=str(table_plan_id), 
                                    key="table_plan_id_input", 
                                    help="Edit the plan_id for this table/category.")
with col2:
    table_header = table_data.get("table_header") or ""
    new_table_header = st.text_input("Table Header", 
                                    value=str(table_header), 
                                    key="table_header_input", 
                                    help="Edit the header for this table/category.")

# ---- Column headers extraction ----
def make_unique_names(headers):
    seen = {}
    unique = []
    for i, h in enumerate(headers):
        base = h if h else f"Column {i+1}"
        name = base
        count = 1
        while name in seen:
            name = f"{base}_{count}"
            count += 1
        seen[name] = True
        unique.append(name)
    return unique

raw = table_data.get("columns")
rows = table_data.get("rows", [])

max_cols = 0
for r in rows:
    if isinstance(r, list):
        max_cols = max(max_cols, len(r))
    elif isinstance(r, dict):
        vals = r.get("values") or []
        max_cols = max(max_cols, 1 + len(vals))

headers_list = []
if raw and isinstance(raw, list) and any(str(h).strip() for h in raw):
    headers_list = [str(h) for h in raw]
elif raw and isinstance(raw, str) and raw.strip():
    headers_list = [h for h in raw.split(",") if h.strip()]
else:
    headers_list = []

if len(headers_list) < max_cols:
    headers_list = headers_list + ["" for _ in range(max_cols - len(headers_list))]

if not headers_list or all(h == "" for h in headers_list):
    if rows and isinstance(rows[0], list):
        headers_list = ["" for _ in range(len(rows[0]))]
    elif rows and isinstance(rows[0], dict):
        key = rows[0].get("key", "")
        vals = rows[0].get("values", [])
        headers_list = ["", *("" for _ in range(len(vals)))]

headers = headers_list[:max_cols] if len(headers_list) > max_cols else headers_list
if len(headers) < max_cols:
    headers = headers + ["" for _ in range(max_cols - len(headers))]

unique_colnames = make_unique_names(headers)

# ---- Editable column headers ----
st.markdown("---")
st.subheader("🖊️ Columns")
headers_df = pd.DataFrame([headers], columns=unique_colnames)
st.caption("Edit the column headers below. These will be used as the first row in your table.")
edited_headers = st.data_editor(
    headers_df,
    num_rows="dynamic",
    use_container_width=True,
    key="header_editor",
)
headers = edited_headers.iloc[0].tolist()

# ---- Build dataframe from rows (AFTER editing headers) ----
orig_rows = table_data.get("rows") or []
row_format = "list"
if len(orig_rows) > 0 and isinstance(orig_rows[0], dict):
    row_format = "dict"

normalized_rows = []
for r in orig_rows:
    if row_format == "list":
        key = r[0] if len(r) > 0 else ""
        vals = r[1:]
    else:
        key = r.get("key", "")
        vals = list(r.get("values") or [])

    expected_vals = len(headers) - 1
    vals = vals[:expected_vals]
    while len(vals) < expected_vals:
        vals.append("")

    row_list = [key] + vals
    normalized_rows.append(row_list)

if len(normalized_rows) == 0:
    df = pd.DataFrame(columns=unique_colnames)
else:
    df = pd.DataFrame(normalized_rows, columns=unique_colnames).fillna("")

# ---- Editable table ----
st.markdown("---")
st.subheader("📋 Table Cells")
st.caption("Edit the table cells below. All changes are instantly reflected in the export preview.")
edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="table_editor",
)

# Map edited_df columns back to edited headers for export
edited_df.columns = headers

# ---- Rebuild rows and metadata ----
rebuilt_rows = []
for _, row in edited_df.iterrows():
    key_cell = row.get(headers[0], "")
    if isinstance(key_cell, pd.Series):
        key_cell = key_cell.iloc[0]
    key_str = "" if (pd.isna(key_cell) or str(key_cell).strip() == "") else str(key_cell)

    values = []
    for h in headers[1:]:
        v = row.get(h, "")
        values.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))

    if key_str == "" and all(v == "" for v in values):
        continue

    if row_format == "list":
        rebuilt_rows.append([key_str] + values)
    else:
        rebuilt_rows.append({"key": key_str, "values": values})

updated_containers = list(containers)
updated_table = dict(table_data)
updated_table["plan_id"] = new_table_plan_id
if "table_header" in table_data:
    updated_table["table_header"] = new_table_header
else:
    updated_table["row_header"] = new_table_header

updated_table["columns"] = headers
updated_table["rows"] = rebuilt_rows
updated_containers[selected_index] = updated_table

updated_data = dict(data)
container_type, category_idx = container_map[selected_index]
if container_type == "table":
    updated_data["table"] = updated_table
elif container_type == "category":
    updated_data.setdefault("table", {})
    if "categories" in updated_data["table"] and isinstance(updated_data["table"]["categories"], list):
        updated_data["table"]["categories"][category_idx] = updated_table
    else:
        updated_data["table"]["categories"] = [updated_table]
elif container_type == "root":
    updated_data = updated_table


if isinstance(updated_data.get("table"), dict) and updated_data["table"].get("plan_id") is not None:
    updated_data["table"]["plan_id"] = updated_containers[0].get("plan_id")

updated_data.setdefault("context_before_table", data.get("context_before_table"))
updated_data.setdefault("context_after_table", data.get("context_after_table"))

# ---- Review & Export ----
st.markdown("---")
st.subheader("✅Review & Export")
with st.expander("👁️ Preview Updated JSON", expanded=False):
    st.json(copy.deepcopy(updated_data))

colA, colB = st.columns([2,1])
with colA:
    default_filename = "updated_table.json"
    if uploaded_file is not None and hasattr(uploaded_file, "name"):
        base = uploaded_file.name.rsplit(".", 1)[0]
        default_filename = f"{base}.updated.json"
    st.download_button(
        label="⬇️ Download updated JSON",
        data=json.dumps(updated_data, indent=2),
        file_name=default_filename,
        mime="application/json",
    )
with colB:
    if st.button("🔄 Reset to Original", help="Reload the original uploaded file and discard all changes."):
        st.experimental_rerun()

