// Dark mode toggle
const darkModeToggle = document.getElementById('darkModeToggle');
darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
});
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}
// Drag & Drop and Preview logic
const dragDropArea = document.getElementById('drag-drop-area');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const uploadForm = document.getElementById('upload-form');
const predictBtn = document.getElementById('predict-btn');
const predictBtnText = document.getElementById('predict-btn-text');
const loadingSpinner = document.getElementById('loading-spinner');
const resetBtn = document.getElementById('reset-btn');
dragDropArea.addEventListener('click', () => fileInput.click());
dragDropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dragDropArea.classList.add('dragover');
});
dragDropArea.addEventListener('dragleave', e => {
    e.preventDefault();
    dragDropArea.classList.remove('dragover');
});
dragDropArea.addEventListener('drop', e => {
    e.preventDefault();
    dragDropArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        showPreviews(fileInput.files);
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        showPreviews(fileInput.files);
    } else {
        clearPreview();
    }
});
function showPreviews(files) {
    previewArea.innerHTML = '';
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const reader = new FileReader();
        reader.onload = e => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'img-fluid rounded m-1';
            img.style.maxHeight = '120px';
            img.style.maxWidth = '120px';
            previewArea.appendChild(img);
            previewArea.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }
}
function clearPreview() {
    previewArea.innerHTML = '';
    previewArea.style.display = 'none';
}
// On form submit, show spinner and let the page reload (Flask will serve a fresh form)
uploadForm.addEventListener('submit', () => {
    predictBtnText.style.display = 'none';
    loadingSpinner.classList.remove('d-none');
    // Let the page reload clear the form and preview
});
// On reset, clear form, preview, and file input, then reload page via /reset
resetBtn.addEventListener('click', () => {
    uploadForm.reset();
    fileInput.value = '';
    clearPreview();
    window.location.href = '/reset';
});
// Model Info Modal AJAX
const modelInfoModal = document.getElementById('modelInfoModal');
modelInfoModal.addEventListener('show.bs.modal', function () {
    const body = document.getElementById('model-info-body');
    body.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div></div>';
    fetch('/model_info').then(r => r.json()).then(info => {
        body.innerHTML = `<h5>${info.title}</h5>
        <p>${info.description}</p>
        <p><b>Model:</b> ${info.model}</p>
        <p><b>Classes:</b> ${info.classes.join(', ')}</p>`;
    });
}); 