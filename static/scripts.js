// Event Listeners for UI Interactions

// Makes the message "Text too short, try a longer text" visible if applicable
document.getElementById('check-button').addEventListener('click', function () {
    document.getElementById('text-to-short').style.display = 'block';
});

// Handle file uploads and process the content from .txt or .docx files
document.getElementById('upload-file').addEventListener('change', handleFileUpload);

// Clear the input field, uploaded file, and reset suggestions with user confirmation
document.getElementById('clear-input-button').addEventListener('click', function () {
    const isConfirmed = confirm("Are you sure you want to clear the input? This action cannot be undone.");
    if (isConfirmed) {
        document.getElementById('text-input').value = ""; 
        document.getElementById("suggestions-list").innerHTML = '<li>No suggestions yet. Enter some text or upload a file and click "Check now".</li>';
        document.getElementById("upload-file").value = "";
        document.getElementById('text-to-short').style.display = 'none';
    }
});

// Apply all suggested text changes to the input field when the "Apply All Suggestions" button is clicked
document.getElementById('apply-suggestions-button').addEventListener('click', applyAllSuggestions);

// Hide the loader animation once the page is fully loaded
window.addEventListener('load', function () {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.style.display = 'none'; // Ensure the loader is hidden after the page has finished loading
    }
});

// File Upload Handlers
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();

        // Handle plain text files (.txt)
        if (file.type === 'text/plain') {
            reader.onload = function (e) {
                document.getElementById('text-input').value = e.target.result;
            };
            reader.readAsText(file);
        } 
        // Handle Microsoft Word documents (.docx)
        else if (file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
            const readerDocx = new FileReader();
            readerDocx.onload = function (e) {
                const arrayBuffer = e.target.result;
                // Use Mammoth.js to extract plain text from the .docx file
                mammoth.extractRawText({ arrayBuffer: arrayBuffer })
                    .then(function (result) {
                        document.getElementById('text-input').value = result.value;
                    })
                    .catch(function () {
                        alert('Error extracting text from .docx file.');
                    });
            };
            readerDocx.readAsArrayBuffer(file);
        } 
        // Handle unsupported file types
        else {
            alert('Unsupported file type. Please upload a .txt or .docx file.');
        }
    }
}

// Suggestions 
/**
 * Apply a single suggestion to the text area.
 * @param {string} originalText 
 * @param {string} newText 
 */
function handleSuggestion(originalText, newText) {
    const textArea = document.getElementById('text-input');
    const currentText = textArea.value;
    const updatedText = currentText.replace(originalText, newText);
    textArea.value = updatedText;
}

// Apply all suggestions from the suggestion bubbles to the text area.
function applyAllSuggestions() {
    const textArea = document.getElementById('text-input');
    const currentText = textArea.value;
    const suggestions = document.querySelectorAll('.bubble'); 

    let updatedText = currentText;

    suggestions.forEach((bubble) => {
        const originalText = bubble.getAttribute('data-original');
        const newText = bubble.getAttribute('data-new');

        if (originalText && newText) {
            const regex = new RegExp(originalText, 'g');
            updatedText = updatedText.replace(regex, newText);
        }
    });

    textArea.value = updatedText;
}

// Display Loader
function showLoader() {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.style.display = 'flex'; 
    }
}
