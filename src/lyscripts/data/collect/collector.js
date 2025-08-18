/**
 * Client-side helper functions for collecting user input through JSONEditor,
 * validating it against a fetched JSON Schema, submitting the validated data
 * to the backend, and presenting a downloadable CSV returned by the server.
 *
 * NOTE: Functionality is intentionally unchanged; only readability and
 * documentation have been improved.
 */

/**
 * Ensure an alert element (used to display validation errors) exists.
 * Creates and appends it if missing.
 *
 * @returns {HTMLDivElement} The existing or newly created alert element.
 */
function ensureAlertExists() {
  let alertElement = document.querySelector('.alert');
  if (!alertElement) {
    alertElement = document.createElement('div');
  }
  alertElement.className = 'alert alert-danger';
  const editorHolder = document.getElementById('editor_holder');
  editorHolder.appendChild(alertElement);
  return alertElement;
}

/**
 * Remove an existing validation alert if present.
 */
function ensureAlertRemoved() {
  const existingAlert = document.querySelector('.alert');
  if (existingAlert) {
    console.log('Clearing existing alert');
    existingAlert.remove();
  }
}

/**
 * Remove an existing download button (if it exists) to avoid duplicates.
 */
function ensureDownloadButtonRemoved() {
  const existingButton = document.getElementById('download_link');
  if (existingButton) {
    console.log('Clearing existing download button');
    existingButton.remove();
  }
}

/**
 * Create (or replace) a download button for a CSV blob returned by the server.
 *
 * @param {Blob} blob - The CSV data blob to make downloadable.
 */
function createDownloadButton(blob) {
  ensureDownloadButtonRemoved();

  const url = window.URL.createObjectURL(blob);
  const downloadLink = document.createElement('a');
  downloadLink.id = 'download_link';
  downloadLink.href = url;
  downloadLink.textContent = 'Download CSV';
  downloadLink.className = 'btn btn-success';
  downloadLink.download = 'lydata_records.csv';

  document.getElementById('editor_holder').appendChild(downloadLink);
  console.log('Download button created:', downloadLink);
}

/**
 * Send validated editor data to the backend for processing. Expects a CSV blob
 * in response which is then exposed via a generated download button.
 *
 * @param {JSONEditor} editor - The JSONEditor instance from which to read data.
 */
async function sendEditorData(editor) {
  const data = editor.getValue();
  console.log('Sending data:', data);

  try {
    const response = await fetch('/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      // Try to parse error details from JSON, fallback to text
      let errorMsg = 'Unknown error';
      try {
        const err = await response.json();
        errorMsg = err.detail || err.message || errorMsg;
      } catch {
        errorMsg = await response.text();
      }
      throw new Error(errorMsg);
    }

    const blob = await response.blob();
    console.log('Received processed data as blob:', blob);
    createDownloadButton(blob);
  } catch (error) {
    ensureDownloadButtonRemoved();
    console.error('Error submitting data:', error);
    const alert = ensureAlertExists();
    alert.textContent = 'Error submitting data: ' + error.message;
    alert.classList.add('alert-danger');
  }
}

/**
 * Validate the editor content. If there are validation errors they are
 * displayed in an alert; otherwise the data is submitted to the backend.
 *
 * @param {JSONEditor} editor - The JSONEditor instance to validate & submit.
 */
function processEditor(editor) {
  const errors = editor.validate();

  if (errors.length) {
    console.error('Validation errors:', errors);
    const alert = ensureAlertExists();
    alert.textContent = 'Validation errors: ' + errors.map(e => e.message).join(', ');
  } else {
    console.log('Data successfully validated');
    ensureAlertRemoved();
    sendEditorData(editor);
  }
}

// Fetch the JSON Schema to initialize the editor
fetch('/schema')
  .then(response => response.json())
  .then(schema => {
    const element = document.getElementById('editor_holder');
    const options = {
      disable_edit_json: true,
      theme: 'bootstrap5',
      iconlib: 'bootstrap',
      object_layout: 'grid',
      disable_properties: true,
      schema: schema
  };
  const editor = new JSONEditor(element, options);

    // Bind the submit button to validation + submission flow
    document.getElementById('submit').addEventListener('click', () => {
      console.log('Submit button clicked');
      processEditor(editor);
    });
  })
  .catch(error => console.error('Error loading schema:', error));
