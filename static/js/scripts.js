document.addEventListener("DOMContentLoaded", () => {
    bindFlashDismiss();
    bindUploadForm();
    updateOptionCounter();
});

function bindFlashDismiss() {
    document.querySelectorAll("[data-flash-dismiss]").forEach((button) => {
        button.addEventListener("click", () => {
            const flash = button.closest(".flash");
            if (flash) {
                flash.remove();
            }
        });
    });
}

function bindUploadForm() {
    const form = document.querySelector("[data-upload-form]");
    if (!form) {
        return;
    }

    const fileInput = form.querySelector("[data-file-input]");
    const fileMeta = form.querySelector("[data-file-meta]");
    const submitButton = form.querySelector("[data-submit-button]");
    const optionInputs = form.querySelectorAll("[data-analysis-option]");

    if (fileInput && fileMeta) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files && fileInput.files[0];
            const maxUploadMb = Number(fileInput.dataset.maxUploadMb || "0");

            if (!file) {
                fileMeta.innerHTML = "<strong>No file selected</strong><span>Select an ArduPilot binary log to continue.</span>";
                if (submitButton) {
                    submitButton.disabled = false;
                }
                return;
            }

            const sizeMb = file.size / (1024 * 1024);
            const tooLarge = maxUploadMb > 0 && sizeMb > maxUploadMb;

            fileMeta.innerHTML = `
                <strong>${escapeHtml(file.name)}</strong>
                <span>${sizeMb.toFixed(2)} MB${tooLarge ? ` - exceeds ${maxUploadMb} MB limit` : ""}</span>
            `;

            if (submitButton) {
                submitButton.disabled = tooLarge;
            }
        });
    }

    optionInputs.forEach((input) => {
        input.addEventListener("change", updateOptionCounter);
    });

    form.addEventListener("submit", () => {
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = "Analyzing...";
        }
    });
}

function updateOptionCounter() {
    const counter = document.querySelector("[data-option-counter]");
    if (!counter) {
        return;
    }

    const checked = document.querySelectorAll("[data-analysis-option]:checked").length;
    counter.textContent = `${checked} view${checked === 1 ? "" : "s"} selected`;
}

function escapeHtml(value) {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
