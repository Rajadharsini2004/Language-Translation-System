
document.getElementById("translateForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const form = document.getElementById("translateForm");
    const formData = new FormData(form);
    const outputDiv = document.getElementById("output");

    outputDiv.innerHTML = "<p>⏳ Translating... Please wait...</p>";

    try {
        const response = await fetch("/translate", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        let html = "";

        // PERFORMANCE METRICS
        if (data.metrics) {
            html += `
                <div class="card">
                    <h3>Performance Metrics</h3>
                    <p>OCR Confidence: ${data.metrics.ocr_confidence !== null ? data.metrics.ocr_confidence + "%" : "N/A"}</p>
                    <p>Translation Time: ${data.metrics.translation_time_sec} sec</p>
                    <p>Total Processing Time: ${data.metrics.total_processing_time_sec} sec</p>
                </div>
            `;
        }

        // TRANSLATION CARDS
        if (data.translations && Object.keys(data.translations).length > 0) {
            for (const lang in data.translations) {
                const acc = data.metrics.translation_accuracy ? data.metrics.translation_accuracy[lang] : null;
                const textAcc = data.metrics.text_input_accuracy ? data.metrics.text_input_accuracy[lang] : null;
                html += `
                    <div class="card">
                        <h3>${lang}</h3>
                        <p>${data.translations[lang]}</p>
                        ${acc !== null ? `<p>Translation Accuracy (BLEU): ${acc}</p>` : ''}
                    </div>
                `;
            }
        } else {
            html += `
                <div class="card">
                    <h3>No Translations</h3>
                    <p>Please select at least one language.</p>
                </div>
            `;
        }

        outputDiv.innerHTML = html;

    } catch (error) {
        console.error(error);
        outputDiv.innerHTML = "<p style='color:red'>❌ Error during translation</p>";
    }
});


