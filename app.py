import streamlit as st
import pandas as pd
import json
from datetime import datetime, timezone
import copy
from typing import List, Dict, Any, Tuple, Optional
import os
from pathlib import Path



def apply_custom_styles() -> None:
    """Apply custom CSS styles to the Streamlit app."""
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


def setup_sidebar() -> None:
    """Set up the sidebar with logo and information."""
    st.sidebar.image("logo/zph-blue.png", width=250)
    st.sidebar.markdown(
        """
        ### 📝 SPD: Table Parsing / Human Feedback
        - Upload your JSON file
        - Select a category
        - Edit headers, plan and table cells
        - Download the updated JSON
        """
    )
    st.sidebar.info("All changes are local until you download the file.")


def setup_page() -> None:
    """Configure the page layout and title."""
    st.set_page_config(page_title="SPD Parser", layout="wide")
    st.title("🧑‍💻 SPD Parser : HITL")
    st.markdown(
        """
        <div style='color: #64748b; font-size: 1.1rem;'>
        Easily Upload Multiple JSON and Edit the Files.
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_file_upload() -> List[Dict]:
    """Handle file upload and return parsed JSON data.
    - If a previously saved version exists in temp/output/json_output, load that instead of the uploaded file so previous edits are preserved.
    - Store a file map in session_state to keep track of the loaded path and original filename.
    """
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

        file_data = []
        output_dir = os.path.join("temp/output", "json_output")
        os.makedirs(output_dir, exist_ok=True)

        # initialize a map of filename -> {path, data}
        if "file_map" not in st.session_state:
            st.session_state["file_map"] = {}

        for uploaded_file in uploaded_files:
            try:
                safe_name = Path(uploaded_file.name).name
                saved_path = os.path.join(output_dir, safe_name)

                if os.path.exists(saved_path):
                    try:
                        with open(saved_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        source = "saved"
                    except Exception:
                        # fallback to uploaded content if saved file cannot be read
                        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                        source = "uploaded"
                else:
                    data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                    source = "uploaded"

                # Save the file to temp/output_json (this will overwrite uploaded on first upload)
                try:
                    tmp_path = saved_path + ".tmp"
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, saved_path)
                except Exception:
                    # if we cannot write the canonical copy, proceed but warn
                    st.warning(f"Could not write canonical copy for {safe_name}")

                # Store the current file path in session state
                st.session_state["current_file_path"] = saved_path

                # also store mapping for reload & lookups
                st.session_state["file_map"][safe_name] = {
                    "path": saved_path,
                    "data": data,
                    "original_filename": uploaded_file.name,
                    "source": source
                }

                file_data.append({
                    "name": safe_name,
                    "data": data,
                    "file": uploaded_file,
                    "path": saved_path
                })
            except Exception as e:
                st.error(f"Failed to parse {uploaded_file.name}: {e}")
        return file_data


def extract_containers(data: Dict) -> Tuple[List[Dict], List[Tuple[str, Optional[int]]], Optional[str]]:
    """Extract table containers from the JSON data."""
    table_block = data.get("table", {})
    containers: List[Dict] = []
    container_map: List[Tuple[str, Optional[int]]] = []
    container_key: Optional[str] = None

    if isinstance(table_block, dict) and table_block.get("Tables") and isinstance(table_block["Tables"], list):
        for idx, tbl in enumerate(table_block["Tables"]):
            containers.append(copy.deepcopy(tbl))
            container_map.append(("Tables", idx))
        container_key = "Tables"
    elif isinstance(table_block, dict):
        containers.append(copy.deepcopy(table_block))
        container_map.append(("table", None))
        container_key = "single"
        if table_block.get("categories") and isinstance(table_block["categories"], list):
            for idx, cat in enumerate(table_block["categories"]):
                if cat is table_block or cat == table_block:
                    continue
                containers.append(copy.deepcopy(cat))
                container_map.append(("category", idx))
            container_key = "categories"

    return containers, container_map, container_key


def make_unique_names(headers: List[str]) -> List[str]:
    """Create unique column names from headers."""
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


def process_headers(table_data: Dict) -> Tuple[List[str], List[str], int]:
    """Process and validate table headers."""
    rows = table_data.get("rows", [])
    raw = table_data.get("column_header") 

    headers_list = [str(h) for h in raw] if raw and isinstance(raw, list) else []

    max_cols = 0
    for r in rows:
        if isinstance(r, dict):
            vals = r.get("values") or []
            max_cols = max(max_cols, 1 + len(vals))
        elif isinstance(r, list):
            max_cols = max(max_cols, len(r))

    if len(headers_list) < max_cols:
        headers_list = headers_list + ["" for _ in range(max_cols - len(headers_list))]
    headers = headers_list[:max_cols] if len(headers_list) > max_cols else headers_list
    if len(headers) < max_cols:
        headers = headers + ["" for _ in range(max_cols - len(headers))]
    
    unique_colnames = make_unique_names(headers)
    return headers, unique_colnames, max_cols


def create_dataframe(table_data: Dict, headers: List[str]) -> pd.DataFrame:
    """Create a pandas DataFrame from table data."""
    orig_rows = table_data.get("rows") or []
    st.session_state["row_format"] = "list"
    if len(orig_rows) > 0 and isinstance(orig_rows[0], dict):
        st.session_state["row_format"] = "dict"
    
    normalized_rows = []
    for r in orig_rows:
        if st.session_state["row_format"] == "list":
            key = r[0] if len(r) > 0 else ""
            vals = r[1:]
            expected_vals = len(headers) - 1
            vals = vals[:expected_vals]
            while len(vals) < expected_vals:
                vals.append("")
            normalized_rows.append([key] + vals)
        else:
            key = r.get("key", "")
            vals = list(r.get("values") or [])
            expected_vals = len(headers) - 1
            vals = vals[:expected_vals]
            while len(vals) < expected_vals:
                vals.append("")
            normalized_rows.append([key] + vals)
    
    if not normalized_rows:
        return pd.DataFrame(columns=make_unique_names(headers))
    return pd.DataFrame(normalized_rows, columns=make_unique_names(headers)).fillna("")


def rebuild_table_data(edited_df: pd.DataFrame, headers: List[str], row_format: str) -> List[Any]:
    """Rebuild table data from edited DataFrame."""
    rebuilt_rows = []
    for _, row in edited_df.iterrows():
        key_cell = row.get(headers[0], "")
        if isinstance(key_cell, pd.Series):
            key_cell = key_cell.iloc[0]
        key_str = "" if (pd.isna(key_cell) or str(key_cell).strip() == "") else str(key_cell)

        values = []
        for h in headers[1:]:  
            v = row.get(h, "")
            # Convert any non-string values to string and handle NaN/None
            if v is None or (isinstance(v, float) and pd.isna(v)):
                values.append("")
            else:
                values.append(str(v))

        # Ensure values array length matches number of columns
        while len(values) < len(headers) - 1: 
            values.append("")

        if row_format == "dict":
            rebuilt_rows.append({"key": key_str, "values": values})
        else:
            rebuilt_rows.append([key_str] + values)
    
    return rebuilt_rows


def handle_table_editing(table_data: Dict) -> Tuple[Dict, List[str]]:
    """Handle the table editing interface."""
    # Edit headers
    st.markdown("---")
    st.subheader("📝 Headers")
    col1, col2 = st.columns([1,1])
    with col1:
        table_plan_name = table_data.get("plan_name") or ""
        new_table_plan_name = st.text_input(
            "Table Plan Name", 
            value=str(table_plan_name), 
            key="table_plan_name_input", 
            help="Edit the plan_name for this table/category."
        )
    with col2:
        table_header = table_data.get("row_header") or ""
        new_table_header = st.text_input(
            "Table Header", 
            value=str(table_header), 
            key="table_header_input", 
            help="Edit the header for this table/category."
        )

    # Process headers and create DataFrame
    headers, unique_colnames, _ = process_headers(table_data)
    
    if "column_header" not in st.session_state:
        st.session_state["column_header"] = headers

    # Column header editing
    st.markdown("---")
    st.subheader("🖊️ Columns")
    add_col = st.button("➕ Add Column", key="add_col_btn", help="Add a new empty column to the table.")

    if add_col and "df" in st.session_state:
        new_col_name = f"Column {len(st.session_state['column_header'])+1}"
        st.session_state["column_header"].append(new_col_name)
        st.session_state["df"][new_col_name] = ""
        st.session_state["df"].columns = make_unique_names(st.session_state["column_header"])
        table_data["column_header"] = st.session_state["column_header"]  

    if "df" not in st.session_state:
        st.session_state["df"] = create_dataframe(table_data, st.session_state["column_header"])

    # Edit headers
    headers_df = pd.DataFrame(
        [st.session_state["column_header"]], 
        columns=make_unique_names(st.session_state["column_header"])
    )
    st.caption("Edit the column headers below. These will be used as the first row in your table.")
    edited_headers = st.data_editor(
        headers_df,
        num_rows="dynamic",
        use_container_width=True,
        key="header_editor",
    )
    st.session_state["column_header"] = edited_headers.iloc[0].tolist()

    st.caption("Edit the table cells below. You can add or remove rows as needed.")
    edited_df = st.data_editor(
        st.session_state["df"],
        num_rows="dynamic",
        use_container_width=True,
        key="table_editor",
    )
    edited_df.columns = st.session_state["column_header"]

    # Update container data
    rebuilt_rows = rebuild_table_data(edited_df, st.session_state["column_header"], st.session_state["row_format"])
    updated_table = dict(table_data)
    updated_table["plan_name"] = new_table_plan_name
    if "row_header" in table_data:
        updated_table["row_header"] = new_table_header
    updated_table["column_header"] = st.session_state["column_header"]
    updated_table["rows"] = rebuilt_rows

    # Update the source table data immediately
    table_data.update(updated_table)

    return updated_table, headers


def create_plan_document(updated_data: Dict, containers: List[Dict], container_key: str) -> Dict:
    """Create the final plan document for export."""
    plan_metadata = {
        "schema_version": "1.0",
        "extraction_model": "ft:gpt4o:zakipoint-health",
        "extracted_at": datetime.now(timezone.utc).isoformat()
    }

    all_tables = []
    if container_key == "Tables":
        for idx, tbl in enumerate(containers):
            table_data = {
                "id": idx,
                "plan_name": tbl.get("plan_name", ""),
                "row_header": tbl.get("row_header", ""),
                "column_header": tbl.get("column_header", []),
                "rows": tbl.get("rows", [])
            }
            # Ensure all rows have the correct number of values based on column headers
            if table_data["rows"] and table_data["column_header"]:
                expected_values = len(table_data["column_header"]) - 1 
                for row in table_data["rows"]:
                    if isinstance(row, dict):
                        while len(row["values"]) < expected_values:
                            row["values"].append("")
                    elif isinstance(row, list):
                        while len(row) < len(table_data["column_header"]):
                            row.append("")
            all_tables.append(table_data)
        plan_name = containers[0].get("plan_name", "") if containers else ""
    else:
        for idx, tbl in enumerate(containers):
            all_tables.append({
                "id": idx,
                "plan_name": tbl.get("plan_name", ""),
                "row_header": tbl.get("row_header", ""),
                "column_header": tbl.get("column_header", []),
                "rows": tbl.get("rows", [])
            })
        plan_name = containers[0].get("plan_name", "")

    if "Tables" in updated_data.get("table", {}):
        return {
            "context_before_table": updated_data.get("context_before_table"),
            "table": {
                "plan_name": plan_name,
                "Tables": all_tables
            },
            "context_after_table": updated_data.get("context_after_table"),
            "metadata": plan_metadata
        }
    elif "categories" in updated_data.get("table", {}):
        plan_document = copy.deepcopy(updated_data)
        plan_document["metadata"] = plan_metadata
        return plan_document
    else:
        plan_document = copy.deepcopy(updated_data)
        plan_document["metadata"] = plan_metadata
        return plan_document

# TODO: Add Ingestion Logic Into Streamlit app
# async def ingest_table_with_mllm(plan_document: Dict) -> None:
#     """
#     Ingest table data using MLLM parser with parse_pdf=False.
#     Args:
#         plan_document: The JSON document containing table data to ingest
#     """
#     temp_dir = os.path.abspath("temp")
#     temp_file = os.path.join(temp_dir, "table_data.json")
#     try:
#         # Setup logging
#         logger = setup_logger("mllm_ingestion")
        
#         # Get LLM instance
#         llm = ChatOpenAI(
#             model="ft:gpt-4o-2024-08-06:zakipoint-health::BoofB2Q5",
#             temperature=0.0,
#             api_key= os.getenv("OPENAI_API_KEY"),
#         )
        
#         # Create temp directory if it doesn't exist
#         os.makedirs(temp_dir, exist_ok=True)
        
#         # Save the plan document
#         with open(temp_file, "w", encoding='utf-8') as f:
#             json.dump(plan_document, f, indent=2, ensure_ascii=False)
        
#         # Process with MLLM parser
#         async with MllmParser.from_defaults(
#             table_file=temp_file,
#             llm=llm,
#             output_dir=os.path.join(temp_dir, "output"),
#             json_out_dir=os.path.join(temp_dir, "json_output"),
#             logger=logger,
#             parse_pdf=False  
#         ) as parser:
#             # Extract table data directly from the JSON
#             table_data = plan_document.get("table", {})
#             if not table_data:
#                 st.error("No table data found in the document")
#                 return
                
#             # Process the table data
#             results = await parser.summarize_json(
#                 file_page_pairs=[(temp_file, 1)],  
#                 json_out_dir=os.path.join(temp_dir, "json_output"),
#                 division_id="temp_div"
#             )
            
#             if results:
#                 st.success("Successfully processed table with MLLM parser")
#                 st.write("Results:")
#                 st.json(results)
#             else:
#                 st.error("No results generated from MLLM parsing")
#     except Exception as e:
#         st.error(f"Error during MLLM processing: {str(e)}")
#         logger.error(f"MLLM processing error: {e}", exc_info=True)
#     finally:
#         # Cleanup temp files
#         try:
#             if os.path.exists(temp_file):
#                 os.remove(temp_file)
#             if os.path.exists(os.path.join(temp_dir, "output")):
#                 shutil.rmtree(os.path.join(temp_dir, "output"))
#             if os.path.exists(os.path.join(temp_dir, "json_output")):
#                 shutil.rmtree(os.path.join(temp_dir, "json_output"))
#         except Exception as e:
#             logger.warning(f"Failed to cleanup temp files: {e}")


def save_json_file(file_path: str, data: Dict) -> None:
    """Save JSON data to a file, overwriting if it exists."""
    try:
        #atomic write to avoid partial files
        tmp_path = file_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)

        st.success(f"Successfully saved to {os.path.basename(file_path)}")
    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")


def handle_export(plan_document: Dict, uploaded_file: Any) -> None:
    """Handle the export interface."""
    st.markdown("---")
    st.subheader("✅Review & Export")

    with st.expander("👁️ Preview Updated JSON", expanded=False):
        st.json(copy.deepcopy(plan_document))

    colA, colB, colC = st.columns([2, 1, 1])
    
    # Save button to overwrite the file in temp/output_json
    with colA:
        if "current_file_path" in st.session_state and st.button("💾 Save Changes", 
            help="Save and overwrite the current file in temp/output_json directory"):
            
            # Ensure we have a valid uploaded file
            if uploaded_file is None or not hasattr(uploaded_file, "name"):
                st.error("No uploaded file available to overwrite.")
            else:
                # Sanitize filename to avoid path traversal and keep only basename.
                safe_name = Path(uploaded_file.name).name 

                # Ensure target directory exists.
                out_dir = os.path.join("temp/output", "json_output")  
                os.makedirs(out_dir, exist_ok=True)  

                save_path = os.path.join(out_dir, safe_name)  
                try:
                    save_json_file(save_path, plan_document)

                    # update session_state and file_map to point to the new saved file
                    st.session_state["current_file_path"] = save_path
                    if "file_map" not in st.session_state:
                        st.session_state["file_map"] = {}
                    st.session_state["file_map"][safe_name] = {
                        "path": save_path,
                        "data": plan_document,
                        "original_filename": uploaded_file.name,
                        "source": "saved"
                    }
                except Exception as e:
                    st.error(f"Failed to save file: {e}")  
    
    with colB:
        default_filename = "updated_table.json"
        if uploaded_file is not None and hasattr(uploaded_file, "name"):
            base = uploaded_file.name.rsplit(".", 1)[0]
            default_filename = f"{base}.updated.json"
        st.download_button(
            label="⬇️ Download updated JSON",
            data=json.dumps(plan_document, indent=2),
            file_name=default_filename,
            mime="application/json",
        )
    
    with colC:
        if st.button("🔄 Ingest with MLLM Parser", key="ingest_mllm"):
            st.write("Processing table with MLLM parser...")
            # asyncio.run(ingest_table_with_mllm(plan_document))

        st.markdown("---")
        if st.button(
            "🔄 Restart JSON (Reload Original)", 
            key="restart_json_btn", 
            help="Reload the original uploaded JSON file. All unsaved changes will be lost."
        ):
            # reload from saved canonical copy if available, else from original uploaded file
            reload_path = None
            if uploaded_file is not None and hasattr(uploaded_file, "name"):
                safe_name = Path(uploaded_file.name).name
                if "file_map" in st.session_state and safe_name in st.session_state["file_map"]:
                    reload_path = st.session_state["file_map"][safe_name].get("path")

            try:
                if reload_path and os.path.exists(reload_path):
                    with open(reload_path, 'r', encoding='utf-8') as f:
                        reloaded = json.load(f)
                    # Replace the current streamlit session values for df/column_header
                    if "df" in st.session_state:
                        del st.session_state["df"]
                    if "column_header" in st.session_state:
                        del st.session_state["column_header"]
                    # Overwrite the uploaded file's data in-place to reflect reload
                    st.experimental_set_query_params()  # no-op to ensure some state-change
                    st.session_state["last_file_idx"] = st.session_state.get("last_file_idx", 0)
                    st.experimental_rerun()
                else:
                    # fallback: simply rerun to reload original uploaded context
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Failed to reload file: {e}")


def reset_session_state(selected_index: int) -> None:
    """Reset session state when category changes."""
    if "last_selected_index" not in st.session_state:
        st.session_state["last_selected_index"] = selected_index
    elif st.session_state["last_selected_index"] != selected_index:
        # Clear the session state for table data when category changes
        if "df" in st.session_state:
            del st.session_state["df"]
        if "column_header" in st.session_state:
            del st.session_state["column_header"]
        st.session_state["last_selected_index"] = selected_index


def update_data_structure(data: Dict, updated_table: Dict, containers: List[Dict], 
                         container_map: List[Tuple[str, Optional[int]]], selected_index: int) -> Dict:
    """Update the data structure with edited table information."""
    updated_containers = list(containers)
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

    if isinstance(updated_data.get("table"), dict) and updated_data["table"].get("plan_name") is not None:
        updated_data["table"]["plan_name"] = updated_containers[0].get("plan_name")

    # Preserve context_before_table / context_after_table
    updated_data.setdefault("context_before_table", data.get("context_before_table"))
    updated_data.setdefault("context_after_table", data.get("context_after_table"))

    return updated_data


def main():
    """Main application function."""
    apply_custom_styles()
    setup_sidebar()
    setup_page()

    file_data = handle_file_upload()

    st.markdown("---")
    st.subheader("✏️ Select a file to edit")
    file_names = [f["name"] for f in file_data]
    
    # Store the last selected file index
    if "last_file_idx" not in st.session_state:
        st.session_state["last_file_idx"] = 0
        
    selected_file_idx = st.selectbox(
        "Choose a file to edit", 
        options=range(len(file_names)), 
        format_func=lambda i: file_names[i],
        key="file_selector"
    )
    
    # Clear session state when file changes
    if st.session_state.get("last_file_idx") != selected_file_idx:
        # Reset all table-related session state
        keys_to_clear = ["df", "column_header", "row_format", "last_selected_index"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state["last_file_idx"] = selected_file_idx
    
    selected_file = file_data[selected_file_idx]

    chosen_name = selected_file["name"]
    if "file_map" in st.session_state and chosen_name in st.session_state["file_map"]:
        data = st.session_state["file_map"][chosen_name].get("data")
        # if file_map path exists and the data on disk might have been updated, try to read it
        disk_path = st.session_state["file_map"][chosen_name].get("path")
        if disk_path and os.path.exists(disk_path):
            try:
                with open(disk_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                # keep in-memory cached data as fallback
                pass
        uploaded_file = selected_file["file"]
    else:
        data = selected_file["data"]
        uploaded_file = selected_file["file"]

    st.success(f"JSON loaded successfully from {uploaded_file.name}.")
    with st.expander("\U0001F50D View Raw JSON", expanded=False):
        st.json(data)

    containers, container_map, container_key = extract_containers(data)
    if not containers:
        st.error("No recognizable table structure found under data['table'] or at the root level.")
        st.stop()

    # Container selection interface
    st.markdown("---")
    st.subheader("\U0001F4C1 Select Category")
    table_block = data.get("table", {})

    container_labels = []
    for idx, c in enumerate(containers):
        if idx == 0 and (c is table_block):
            label = "Top Level Table"
            p_name = c.get("plan_name") or ""
            th = c.get("row_header") or ""
            if p_name or th:
                label += f" (plan_name={p_name} | header={th})"
            container_labels.append(label)
        else:
            p_name = c.get("plan_name") or "N/A"
            th = c.get("row_header") or ""
            container_labels.append(f"{idx+1}: plan_name={p_name} | header={th}")

    selected_index = st.selectbox(
        "Select Category",
        options=list(range(len(containers))),
        format_func=lambda i: container_labels[i],
        index=0,
        key="container_selector",
        help="Choose which table/category to edit."
    )

    reset_session_state(selected_index)
    table_data = containers[selected_index]
    updated_table, headers = handle_table_editing(table_data)

    # Update data structure
    updated_data = update_data_structure(data, updated_table, containers, container_map, selected_index)
    plan_document = create_plan_document(updated_data, containers, container_key)
    handle_export(plan_document, uploaded_file)

if __name__ == "__main__":
    main()
