// Interaksi UI halaman upload, drag and drop, dan preview gambar

document.addEventListener("DOMContentLoaded", function () {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fruitImageInput");
    const previewImage = document.getElementById("previewImage");
    const placeholder = document.getElementById("dropzonePlaceholder");
    const fileNameLabel = document.getElementById("fileNameLabel");
    const fileNameText = document.getElementById("fileNameText");
    const clearFileBtn = document.getElementById("clearFileBtn");
    const uploadForm = document.getElementById("uploadForm");
    const submitBtn = document.getElementById("submitBtn");

    if (!dropzone || !fileInput) return;

    // Menampilkan preview gambar terpilih di dalam dropzone
    function showPreview(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (event) {
            previewImage.src = event.target.result;
            previewImage.classList.remove("d-none");
            placeholder.classList.add("d-none");
        };
        reader.readAsDataURL(file);

        fileNameText.textContent = file.name;
        fileNameLabel.classList.remove("d-none");
    }

    // Mengembalikan dropzone ke kondisi kosong
    function resetDropzone() {
        fileInput.value = "";
        previewImage.src = "";
        previewImage.classList.add("d-none");
        placeholder.classList.remove("d-none");
        fileNameLabel.classList.add("d-none");
    }

    // File dipilih lewat file picker
    fileInput.addEventListener("change", function () {
        if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
    });

    // Highlight dropzone saat file diseret masuk
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    // Hapus highlight saat file keluar atau dilepas
    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    // File di-drop ke dropzone
    dropzone.addEventListener("drop", function (e) {
        const droppedFiles = e.dataTransfer.files;
        if (droppedFiles && droppedFiles[0]) {
            fileInput.files = droppedFiles;
            showPreview(droppedFiles[0]);
        }
    });

    // Tombol hapus file yang sudah dipilih
    if (clearFileBtn) {
        clearFileBtn.addEventListener("click", function () {
            resetDropzone();
        });
    }

    // Tampilkan status loading saat form disubmit
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener("submit", function () {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Menganalisis...';
        });
    }
});
