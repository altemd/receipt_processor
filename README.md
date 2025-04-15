# receipt_processor
Gradio app that facilitates the scanning, parsing, and organizing of receipt images

Initially uses PaddleOCR to scan and transcribe the image.
The transcription is passed to a local model (primarily Phi-4 Mini or Gemma 3 4B) to parse through and output the necessary details
The UI places each image and the corresponding outputs for view to the user for final overview and correction
The final output is a CSV file adhering to the import structure of the Cashew Budget App
