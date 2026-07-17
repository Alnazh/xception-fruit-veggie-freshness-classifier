// Rendering semua grafik memakai Chart.js, dipisah dari script.js interaksi UI

const CHART_COLORS = {
    fresh: "#10B981",
    freshSoft: "rgba(16, 185, 129, 0.2)",
    rotten: "#E11D48",
    rottenSoft: "rgba(225, 29, 72, 0.2)",
    neutral: "#F59E0B",
    grid: "rgba(76, 29, 149, 0.07)",
    text: "#6B6280",
};

const BASE_CHART_OPTIONS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: CHART_COLORS.text, font: { family: "Inter" } } },
    },
    scales: {
        x: { ticks: { color: CHART_COLORS.text }, grid: { color: CHART_COLORS.grid } },
        y: { ticks: { color: CHART_COLORS.text }, grid: { color: CHART_COLORS.grid } },
    },
};

// Bar chart horizontal top 5 probabilitas tertinggi, dipakai di halaman klasifikasi
window.renderProbabilityChart = function (canvasId, probData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !probData) return;

    const top5 = [...probData].sort((a, b) => b.probability - a.probability).slice(0, 5);
    const labels = top5.map((item) => item.display);
    const values = top5.map((item) => item.probability);
    const colors = top5.map((item) => (item.condition === "fresh" ? CHART_COLORS.fresh : CHART_COLORS.rotten));

    // tinggi tetap karena bar dibatasi maksimal 5 baris
    const container = canvas.parentElement;
    if (container) container.style.height = Math.max(220, labels.length * 44) + "px";

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: { labels: labels, datasets: [{ label: "Probabilitas (%)", data: values, backgroundColor: colors, borderRadius: 6 }] },
        options: {
            ...BASE_CHART_OPTIONS,
            indexAxis: "y",
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.x}%` } },
            },
            scales: {
                x: { beginAtZero: true, max: 100, ticks: { color: CHART_COLORS.text, callback: (v) => v + "%" }, grid: { color: CHART_COLORS.grid } },
                y: { ticks: { color: CHART_COLORS.text, autoSkip: false }, grid: { display: false } },
            },
        },
    });
};

// Empat line chart akurasi dan loss tahap 1 dan tahap 2, dipakai di halaman model
window.renderTrainingCharts = function (metrics) {
    if (!metrics) return;

    function makeEpochLabels(historyObj) {
        return historyObj.accuracy.map((_, i) => `Epoch ${i + 1}`);
    }

    function renderLineChart(canvasId, labels, trainData, valData, trainLabel, valLabel, isPercent) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        new Chart(canvas.getContext("2d"), {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { label: trainLabel, data: trainData, borderColor: CHART_COLORS.fresh, backgroundColor: CHART_COLORS.freshSoft, tension: 0.3, fill: true },
                    { label: valLabel, data: valData, borderColor: CHART_COLORS.rotten, backgroundColor: CHART_COLORS.rottenSoft, tension: 0.3, fill: true },
                ],
            },
            options: {
                ...BASE_CHART_OPTIONS,
                scales: {
                    ...BASE_CHART_OPTIONS.scales,
                    y: { ...BASE_CHART_OPTIONS.scales.y, ticks: { color: CHART_COLORS.text, callback: (v) => (isPercent ? Math.round(v * 100) + "%" : v) } },
                },
            },
        });
    }

    const stage1Labels = makeEpochLabels(metrics.history_stage1);
    renderLineChart("stage1AccuracyChart", stage1Labels, metrics.history_stage1.accuracy, metrics.history_stage1.val_accuracy, "Train Accuracy", "Validation Accuracy", true);
    renderLineChart("stage1LossChart", stage1Labels, metrics.history_stage1.loss, metrics.history_stage1.val_loss, "Train Loss", "Validation Loss", false);

    const stage2Labels = makeEpochLabels(metrics.history_stage2);
    renderLineChart("stage2AccuracyChart", stage2Labels, metrics.history_stage2.accuracy, metrics.history_stage2.val_accuracy, "Train Accuracy", "Validation Accuracy", true);
    renderLineChart("stage2LossChart", stage2Labels, metrics.history_stage2.loss, metrics.history_stage2.val_loss, "Train Loss", "Validation Loss", false);
};

// Grouped bar chart komposisi dataset train vs test per kelas, dipakai di halaman dataset
window.renderDatasetChart = function (canvasId, datasetCounts) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !datasetCounts) return;

    const classNames = Object.keys(datasetCounts.train || {});
    const trainValues = classNames.map((name) => datasetCounts.train[name] || 0);
    const testValues = classNames.map((name) => datasetCounts.test[name] || 0);

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: classNames,
            datasets: [
                { label: "Data Training", data: trainValues, backgroundColor: CHART_COLORS.fresh, borderRadius: 6 },
                { label: "Data Uji", data: testValues, backgroundColor: CHART_COLORS.neutral, borderRadius: 6 },
            ],
        },
        options: BASE_CHART_OPTIONS,
    });
};
