import gradio as gr
import pandas as pd
from paddleocr import PaddleOCR
from openai import OpenAI # Keep using the openai library
import os
import datetime
import json
import io
import csv
import tempfile
from PIL import Image
import sys # To check for environment variables

# --- Configuration ---

# --- LLM Configuration ---
# Check for local LLM configuration first
LOCAL_LLM_URL = "http://localhost:5001/v1"
LOCAL_LLM_MODEL_NAME = "phi4mini" # e.g., "llama3", "mistral", "phi3" - MUST match the model served by your local server
OPENAI_API_KEY = "local"

client = None
llm_model_to_use = ""
llm_config_source = ""

# --- Default LLM Prompt ---
DEFAULT_LLM_PROMPT_TEMPLATE = """
Extract the key details from the following receipt text. Structure the output strictly as a JSON object with these exact keys: "Date", "Amount", "Category", "Title", "Note", "Account".

- Date: Find the transaction date (format as YYYY-MM-DD HH:MM:SS if possible, otherwise keep original). If no date, use "N/A".
- Amount: Find the final total amount paid (numeric value only, no currency symbols). If no amount, use "N/A".
- Category: Suggest a likely expense category (e.g., Dining, Groceries, Shopping, Transit, Entertainment, Bills & Fees, Gifts, Beauty, Work, Travel, Balance Correction, Income, Education, Other). If unsure, use "Other".
- Title: Extract the product name or a brief description of the purchase (e.g., "Siopao", "Skittles", "Milktea"). If no title, use "N/A".
- Note: Include any other potentially relevant details like the flavor and the address.
- Account: Choose between Cash or Bank. If it is a bank transaction, choose Bank, otherwise use Cash. If unknown, use "N/A".

Example Output:

- Date: 1/7/2025  12:25:42
- Amount: 43
- Category: Dining
- Title: Siopao
- Note: Asado, 7-11, Quezon City
- Account: Cash

IMPORTANT: Respond ONLY with the JSON object, nothing else before or after it.

Receipt Text:
\"\"\"
{text}
\"\"\"

JSON Output:
"""

if LOCAL_LLM_URL:
    print(f"--- Attempting to connect to Local LLM via: {LOCAL_LLM_URL} ---")
    if not LOCAL_LLM_MODEL_NAME:
        print("Warning: LOCAL_LLM_URL is set, but LOCAL_LLM_MODEL is not. Please set LOCAL_LLM_MODEL to the name of the model served locally.")
        # Provide a default or exit? Let's provide a placeholder, user must know their model
        LOCAL_LLM_MODEL_NAME = "local-model" # Placeholder
        print(f"Using placeholder model name: {LOCAL_LLM_MODEL_NAME}")

    try:
        client = OpenAI(
            base_url=LOCAL_LLM_URL,
            api_key="nokey" # Local servers typically don't need a key, "nokey" is common practice
        )
        # Perform a quick test call (optional but recommended)
        client.models.list() # This might fail if the local server's /models endpoint differs
        print(f"Successfully initialized client for Local LLM at {LOCAL_LLM_URL}")
        llm_model_to_use = LOCAL_LLM_MODEL_NAME
        llm_config_source = f"Local LLM ({llm_model_to_use} at {LOCAL_LLM_URL})"
    except Exception as e:
        print(f"Error connecting to Local LLM at {LOCAL_LLM_URL}: {e}")
        print("Please ensure the local server is running and the URL is correct.")
        client = None # Indicate failure

# Fallback to OpenAI if local connection wasn't attempted or failed, and API key exists
if client is None and OPENAI_API_KEY:
    print("--- Attempting to connect to OpenAI API ---")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Test connection
        client.models.list()
        print("Successfully initialized client for OpenAI API.")
        llm_model_to_use = "gpt-3.5-turbo" # Default OpenAI model
        llm_config_source = f"OpenAI API (Model: {llm_model_to_use})"
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Please ensure the OPENAI_API_KEY environment variable is set correctly.")
        client = None # Indicate failure

# If neither worked
if client is None:
     print("------------------------------------------------------------")
     print("FATAL: Could not initialize any LLM client.")
     print("Please configure either:")
     print("  1. Local LLM: Set LOCAL_LLM_URL and LOCAL_LLM_MODEL environment variables.")
     print("  2. OpenAI API: Set the OPENAI_API_KEY environment variable.")
     print("------------------------------------------------------------")
     # Exit or let Gradio show an error? Let Gradio show error message below.


# Initialize PaddleOCR (same as before)
print("Initializing PaddleOCR... (This might take a moment on first run)")
try:
    ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    print("PaddleOCR initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
    print("Please ensure PaddleOCR and its dependencies are installed correctly.")
    ocr_reader = None

# Define CSV columns
CSV_COLUMNS = ["Date", "Amount", "Category", "Title", "Note", "Account", "Original_Filename"]

# --- Helper Functions ---

def run_ocr(image_path):
    """Runs PaddleOCR on a given image path and returns extracted text."""
    if not ocr_reader:
        return "ERROR: PaddleOCR not initialized."
    if not os.path.exists(image_path):
        return "ERROR: Image file not found."
    try:
        result = ocr_reader.ocr(image_path, cls=True)
        lines = []
        if result and result[0]:
             for idx in range(len(result[0])):
                res = result[0][idx]
                lines.append(res[1][0])
        return "\n".join(lines)
    except Exception as e:
        print(f"Error during OCR processing for {image_path}: {e}")
        return f"ERROR: OCR failed - {e}"

def extract_details_with_llm(text, prompt_template): # Added prompt_template argument
    """Uses configured LLM (OpenAI or Local) to extract structured details using the provided prompt template."""
    if not client:
        return {"error": "LLM client not initialized."}
    if not text or text.startswith("ERROR:"):
         return {
            "Date": "N/A", "Amount": "N/A", "Category": "N/A",
            "Title": "OCR Failed or Empty", "Note": text, "Account": "N/A"
         }

    # Use the passed prompt_template. Ensure the template includes "{text}" placeholder.
    try:
        # Format the template with the actual OCR text
        prompt = prompt_template.format(text=text)
    except KeyError:
        # Handle case where the user's template is missing the {text} placeholder
        print("ERROR: Prompt template is missing the required '{text}' placeholder.")
        return {"error": "Prompt template error: Missing '{text}' placeholder."}
    except Exception as e:
        print(f"ERROR: Error formatting prompt template: {e}")
        return {"error": f"Prompt template formatting error: {e}"}


    try:
        print(f"Sending request to LLM ({llm_config_source}) with model '{llm_model_to_use}'")
        # print(f"Using Prompt:\n{prompt[:500]}...") # Optional: Log part of the formatted prompt
        response = client.chat.completions.create(
            model=llm_model_to_use,
            messages=[
                # Adjust system prompt if needed, or remove if your template handles everything
                {"role": "system", "content": "You are an assistant that extracts information from receipts and outputs JSON based on user instructions."},
                {"role": "user", "content": prompt} # Use the formatted prompt
            ],
            temperature=0.1,
            # response_format={"type": "json_object"} # Add back if needed and supported
        )
        content = response.choices[0].message.content
        print(f"Raw LLM response: {content}")

        # --- Rest of the function (JSON parsing, validation) remains the same ---
        try:
            # Find the start and end of the JSON object
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 json_str = content[json_start:json_end]
                 extracted_data = json.loads(json_str)
            else:
                # Fallback: maybe the model returned ONLY JSON? Try parsing the whole thing
                try:
                    extracted_data = json.loads(content)
                except json.JSONDecodeError:
                     raise json.JSONDecodeError("Could not find JSON object or parse response", content, 0)

        except json.JSONDecodeError as e:
             print(f"Error decoding JSON response from LLM: {e}")
             print(f"Raw response content that failed parsing: {content}")
             return {"error": f"LLM returned non-JSON or invalid JSON: {content[:200]}..."}

        for key in ["Date", "Amount", "Category", "Title", "Note", "Account"]:
            extracted_data.setdefault(key, "N/A")

        if isinstance(extracted_data.get("Amount"), str):
             cleaned_amount = extracted_data["Amount"].replace('$','').replace(',','').strip()
             try:
                 float(cleaned_amount)
                 extracted_data["Amount"] = cleaned_amount
             except ValueError:
                 extracted_data["Amount"] = cleaned_amount

        return extracted_data

    except Exception as e:
        print(f"Error calling LLM API ({llm_config_source}): {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"LLM API call failed: {e}"}


# --- Gradio Specific Functions --- (Keep these mostly the same)

def process_all_images_background(files, current_state, prompt_template): # Added prompt_template argument
    """
    Handles upload, runs OCR & LLM for all images in the background,
    and prepares the state for verification. Uses the provided prompt template.
    """
    # --- Keep the initial checks and UI updates the same ---
    if not files:
        # ... (return statement for no files)
        return { ... } # Keep existing return structure

    print("--- Starting Background Processing ---")
    initial_updates = {
        upload_button: gr.update(visible=False),
        start_button: gr.update(visible=False),
        prompt_textbox: gr.update(interactive=False), # Disable prompt editing during processing
        progress_label: gr.update(value="Processing all receipts (OCR & LLM)... This may take a while.", visible=True),
        verification_box: gr.update(visible=False)
    }

    # --- Loop through images (keep this part) ---
    image_paths = [file.name for file in files]
    original_filenames = [os.path.basename(file.name) if file.name else f"receipt_{i+1}.png" for i, file in enumerate(files)]
    total_images = len(image_paths)
    preprocessed_results = []

    for idx, (img_path, filename) in enumerate(zip(image_paths, original_filenames)):
        print(f"Background processing: Image {idx + 1}/{total_images} ({filename})")
        # Update progress label if desired...

        # 1. Run OCR (keep this)
        ocr_text = run_ocr(img_path)

        extracted_details = {} # Initialize details dict
        if ocr_text.startswith("ERROR:"):
            print(f"OCR Error on {filename}: {ocr_text}")
            extracted_details = { ... } # Keep existing OCR error handling
        else:
            # 2. Extract with LLM - PASS THE PROMPT TEMPLATE HERE
            extracted_details = extract_details_with_llm(ocr_text, prompt_template) # Pass it here

            if "error" in extracted_details:
                print(f"LLM Error on {filename}: {extracted_details['error']}")
                # Keep existing LLM error handling
                note_val = extracted_details["error"]
                extracted_details = {k: "LLM Error" for k in CSV_COLUMNS if k not in ["Note", "Original_Filename"]}
                extracted_details["Note"] = note_val
            else:
                extracted_details["Amount"] = str(extracted_details.get("Amount", ""))


        # Store result (keep this)
        preprocessed_results.append({
            "image_path": img_path,
            "original_filename": filename,
            "ocr_text": ocr_text,
            "extracted_details": extracted_details
        })

    print("--- Background Processing Complete ---")

    # Update state (keep this)
    current_state = {
        "preprocessed_results": preprocessed_results,
        "processed_data": [],
        "current_index": 0
    }

    # --- Display first result or handle no results (keep this logic) ---
    if not preprocessed_results:
        print("No images were successfully preprocessed.")
        # Also re-enable prompt textbox on failure
        no_results_updates = { ... } # Existing dictionary for no results
        no_results_updates[prompt_textbox] = gr.update(interactive=True) # Re-enable prompt
        return no_results_updates
    else:
        first_display_updates = display_next_preprocessed_result(current_state)
        final_updates = {**initial_updates, **first_display_updates}
        final_updates[state] = current_state
        # Re-enable prompt editing AFTER processing is done and first item displayed
        final_updates[prompt_textbox] = gr.update(interactive=True)
        # Ensure verification box visibility is handled correctly by display_next_preprocessed_result
        return final_updates


def display_next_preprocessed_result(current_state):
    """Displays the pre-processed results for the current index."""
    idx = current_state["current_index"]
    total_images = len(current_state["preprocessed_results"])

    if idx >= total_images:
        # All images verified, finalize
        # Note: finalize_processing might need the state passed to it if it doesn't take it already
        return finalize_processing(current_state)

    # Retrieve pre-processed data
    data = current_state["preprocessed_results"][idx]
    image_path = data["image_path"]
    original_filename = data["original_filename"]
    ocr_text = data["ocr_text"]
    extracted_details = data["extracted_details"]

    print(f"Displaying pre-processed data for verification: Image {idx + 1}/{total_images} ({original_filename})")

    # Prepare updates for Gradio components
    progress_text = f"Verifying Receipt {idx + 1} of {total_images} ({original_filename})"

    # Ensure Note field gets the potential error message if present
    note_val = extracted_details.get("Note", "")
    if extracted_details.get("error") and "Note" not in extracted_details : # If error happened but wasn't stored in Note
        note_val = extracted_details.get("error", "Unknown error during preprocessing.")

    # Ensure amount is string
    amount_val = str(extracted_details.get("Amount", ""))


    return {
        # state: current_state, # Don't return state here, let the calling function handle it
        image_output: gr.update(value=Image.open(image_path), visible=True),
        ocr_output: gr.update(value=ocr_text, visible=True),
        date_field: gr.update(value=extracted_details.get("Date", ""), visible=True),
        amount_field: gr.update(value=amount_val, visible=True),
        category_field: gr.update(value=extracted_details.get("Category", ""), visible=True),
        title_field: gr.update(value=extracted_details.get("Title", ""), visible=True),
        note_field: gr.update(value=note_val, visible=True),
        account_field: gr.update(value=extracted_details.get("Account", ""), visible=True),
        verify_button: gr.update(visible=True),
        progress_label: gr.update(value=progress_text, visible=True),
        # Ensure these stay hidden/visible correctly
        upload_button: gr.update(visible=False),
        start_button: gr.update(visible=False),
        download_button: gr.update(visible=False),
        final_message: gr.update(value=""),
        verification_box: gr.update(visible=True) # Make sure verification box is visible
    }


def verify_and_process_next(current_state, date, amount, category, title, note, account):
    """Saves the verified/edited data and displays the next pre-processed receipt."""
    idx = current_state["current_index"]
    total_images = len(current_state["preprocessed_results"])

    if idx >= total_images:
         print("Warning: verify_and_process_next called when already past the end.")
         # Should ideally not happen if UI is correct, maybe finalize again?
         updates = finalize_processing(current_state)
         updates[state] = current_state # Ensure state is returned
         return updates


    # Retrieve original filename from the preprocessed data for this index
    original_filename = current_state["preprocessed_results"][idx]["original_filename"]

    verified_data = {
        "Date": date,
        "Amount": amount, # Amount is taken directly from user input field
        "Category": category,
        "Title": title,
        "Note": note,
        "Account": account,
        "Original_Filename": original_filename
    }
    print(f"Verified data for {original_filename}: {verified_data}")

    # Append verified data to the final list
    current_state["processed_data"].append(verified_data)
    # Increment index for the next item
    current_state["current_index"] += 1

    # Get UI updates for the next item (or finalize if done)
    next_display_updates = display_next_preprocessed_result(current_state)

    # Important: Return the updated state along with the UI updates
    next_display_updates[state] = current_state
    return next_display_updates 


def finalize_processing(current_state):
    """Generates the CSV file and provides the download button."""
    print("--- Finalizing Processing ---")
    if not current_state.get("processed_data"): # Use .get for safety
         print("No data was verified.")
         return {
            state: current_state, # Keep state
            image_output: gr.update(visible=False),
            ocr_output: gr.update(value="No data verified.", visible=True), # Show message here
            date_field: gr.update(visible=False), amount_field: gr.update(visible=False), category_field: gr.update(visible=False),
            title_field: gr.update(visible=False), note_field: gr.update(visible=False), account_field: gr.update(visible=False),
            verify_button: gr.update(visible=False),
            progress_label: gr.update(value="Processing complete.", visible=True),
            download_button: gr.update(visible=False),
            final_message: gr.update(value="No data was verified to download."),
            verification_box: gr.update(visible=False), # Hide verification box
            upload_button: gr.update(visible=True), # Allow new uploads
            start_button: gr.update(visible=True) # Allow starting again
        }

    df = pd.DataFrame(current_state["processed_data"], columns=CSV_COLUMNS)
    csv_file_path = "" # Initialize path variable

    try:
        # Create a temporary file for the CSV
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8', newline='') as temp_csv:
            df.to_csv(temp_csv.name, index=False, quoting=csv.QUOTE_ALL)
            csv_file_path = temp_csv.name
        print(f"Generated CSV file at: {csv_file_path}")
        completion_message = f"Processed {len(current_state['processed_data'])} receipts. Ready to download."

        return {
            state: current_state, # Keep state
            image_output: gr.update(visible=False),
            ocr_output: gr.update(visible=False),
            date_field: gr.update(visible=False), amount_field: gr.update(visible=False), category_field: gr.update(visible=False),
            title_field: gr.update(visible=False), note_field: gr.update(visible=False), account_field: gr.update(visible=False),
            verify_button: gr.update(visible=False),
            progress_label: gr.update(value="Processing Complete!", visible=True),
            download_button: gr.update(value=csv_file_path, visible=True), # Provide path to button
            final_message: gr.update(value=completion_message),
            verification_box: gr.update(visible=False), # Hide verification box
            upload_button: gr.update(visible=True), # Allow new uploads
            start_button: gr.update(visible=True) # Allow starting again
        }
    except Exception as e:
        print(f"Error generating or saving CSV: {e}")
        return {
            state: current_state, # Keep state
            image_output: gr.update(visible=False),
            ocr_output: gr.update(value=f"Error creating CSV: {e}", visible=True), # Show error
            date_field: gr.update(visible=False), amount_field: gr.update(visible=False), category_field: gr.update(visible=False),
            title_field: gr.update(visible=False), note_field: gr.update(visible=False), account_field: gr.update(visible=False),
            verify_button: gr.update(visible=False),
            progress_label: gr.update(value="Error", visible=True),
            download_button: gr.update(visible=False),
            final_message: gr.update(value="Failed to generate CSV file."),
            verification_box: gr.update(visible=False),
            upload_button: gr.update(visible=True),
            start_button: gr.update(visible=True)
        }

# --- Gradio Interface ---

css = """
#verification_row{ align-items: flex-start; }
#detail_col{ border: 1px solid #E5E7EB; padding: 15px; border-radius: 8px; margin-left:10px}
#output_area{ margin-top: 20px; }
.gr-button { margin: 5px 0; }
#prompt_box label { font-weight: bold; margin-bottom: 5px; }
"""

with gr.Blocks(css=css) as demo:
    # State variable (as defined in step 1)
    state = gr.State({
        "preprocessed_results": [],
        "processed_data": [],
        "current_index": 0
    })

    gr.Markdown("# Receipt Transcription and Formatting Tool")
    gr.Markdown("Upload receipt images, verify extracted details, and download as CSV.")

    # Display LLM/OCR status (keep this part)
    if llm_config_source: gr.Markdown(f"✅ **LLM Configured:** Using {llm_config_source}")
    else: gr.Markdown(...) # Keep the error message display
    if not ocr_reader: gr.Markdown(...) # Keep the error message display
    core_components_ready = client is not None and ocr_reader is not None
    
    # --- ADD PROMPT TEXTBOX HERE ---
    with gr.Accordion("LLM Prompt Configuration", open=False): # Collapsible section
         gr.Markdown("Edit the prompt template below. Ensure it includes `{text}` where the OCR output should be inserted.")
         prompt_textbox = gr.Textbox(
             label="LLM Prompt Template",
             value=DEFAULT_LLM_PROMPT_TEMPLATE, # Set default value
             lines=15,
             interactive=True, # Allow editing
             elem_id="prompt_box"
         )

    # --- Upload Area ---
    with gr.Row():
        with gr.Column(scale=1):
            upload_button = gr.File(
                label="1. Upload Receipt Images", file_count="multiple", file_types=["image"],
                interactive=core_components_ready
            )
            start_button = gr.Button("2. Start Processing All", variant="primary", visible=True, interactive=core_components_ready)
            progress_label = gr.Markdown("", visible=False) # Shows overall progress / current status

    # --- Verification Area (Initially Hidden) ---
    with gr.Group(visible=False) as verification_box:
        gr.Markdown("### 3. Verify Extracted Details")
        with gr.Row(elem_id="verification_row"): # NEW: Row for side-by-side layout
            with gr.Column(scale=1): # Left column for image
                 image_output = gr.Image(label="Current Receipt Image", type="pil", height=400) # Adjust height as needed
            with gr.Column(scale=1, elem_id="detail_col"): # Right column for details + controls
                 ocr_output = gr.Textbox(label="Raw OCR Text (for reference)", lines=4, interactive=False)
                 with gr.Row():
                     date_field = gr.Textbox(label="Date")
                     amount_field = gr.Textbox(label="Amount")
                 with gr.Row():
                     category_field = gr.Textbox(label="Category")
                     title_field = gr.Textbox(label="Title/Merchant")
                 account_field = gr.Textbox(label="Account/Payment Method")
                 note_field = gr.Textbox(label="Note", lines=3)
                 verify_button = gr.Button("Confirm and Next Receipt ->", variant="secondary")


    # --- Output Area ---
    with gr.Row(visible=True, elem_id="output_area") as output_area:
        final_message = gr.Markdown("")
        download_button = gr.DownloadButton(label="4. Download Results as CSV", visible=False)


    # --- Component Interactions ---

    # Start button triggers background processing of ALL images
    start_button.click(
        fn=process_all_images_background,
        # Add prompt_textbox to the inputs list
        inputs=[upload_button, state, prompt_textbox], # ADDED prompt_textbox
        outputs=[
            # Add prompt_textbox to outputs list as its 'interactive' state changes
            state, image_output, ocr_output, date_field, amount_field, category_field,
            title_field, note_field, account_field, verify_button, progress_label,
            download_button, final_message, verification_box,
            upload_button, start_button, prompt_textbox # ADDED prompt_textbox
        ],
        api_name="start_processing"
    )

    # Verify button saves current, displays next PRE-PROCESSED result
    verify_button.click(
        fn=verify_and_process_next,
        inputs=[state, date_field, amount_field, category_field, title_field, note_field, account_field],
        outputs=[
            state, image_output, ocr_output, date_field, amount_field, category_field,
            title_field, note_field, account_field, verify_button, progress_label,
            download_button, final_message, verification_box,
            upload_button, start_button, prompt_textbox # ADDED prompt_textbox
        ],
        api_name="verify_receipt"
    )

# --- Launch the App ---
if __name__ == "__main__":
    print("--- Preparing to Launch Gradio App ---")
    if core_components_ready:
         print(f"App ready. Configured LLM: {llm_config_source}")
         demo.launch()
    else:
         print("\nApplication cannot launch due to initialization errors (LLM or PaddleOCR).")
         print("Please check the console output above for details and ensure configuration is correct.")
         # Optionally launch a minimal app just showing the error messages from above:
         with gr.Blocks() as error_demo:
             gr.Markdown("# Application Startup Error")
             if not client: gr.Markdown("LLM Client Error: Check configuration (LOCAL_LLM_URL/MODEL or OPENAI_API_KEY) and logs.")
             if not ocr_reader: gr.Markdown("PaddleOCR Error: Check installation and logs.")
         print("Launching minimalist error display.")
         error_demo.launch()