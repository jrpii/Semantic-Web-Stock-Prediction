const paths = {
  modelResults: "../../EVALUATIONS/analysis_outputs/reports/model_results.csv",
  lstm: "../../EVALUATIONS/analysis_outputs/reports/lstm_interval_results.csv",
  lstmNews: "../../EVALUATIONS/analysis_outputs/reports/lstm_news_interval_results.csv",
  lstmFinbert: "../../EVALUATIONS/analysis_outputs/reports/lstm_finbert_interval_results_ExNeEn.csv",
  stockSummary: "../../EVALUATIONS/analysis_outputs/reports/stock_file_summary.csv",
  splits: "../../EVALUATIONS/analysis_outputs/reports/chronological_splits.csv",
  newsDaily: "../../EVALUATIONS/analysis_outputs/cleaned/news/news_daily_counts.csv",
  finbert: "../../EVALUATIONS/analysis_outputs/cleaned/news/news_finbert_by_bar.csv"
};

const reportPaths = [
  ["EDA", "../../EVALUATIONS/analysis_outputs/reports/eda_summary.md"],
  ["Models", "../../EVALUATIONS/analysis_outputs/reports/model_comparison.md"]
];

const colors = ["#1f6d59", "#2f5e95", "#be7d25", "#d56f55", "#6d5c9f", "#0e7c86"];
const metricLabels = {
  roc_auc: "ROC AUC",
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  accuracy: "Accuracy"
};

const state = {
  metric: "roc_auc",
  dataset: "ALL",
  data: null
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(value);
      if (row.some((item) => item.length)) rows.push(row);
      row = [];
      value = "";
    } else {
      value += char;
    }
  }
  if (value.length || row.length) {
    row.push(value);
    rows.push(row);
  }
  const headers = rows.shift() || [];
  return rows.map((items) => Object.fromEntries(headers.map((header, index) => [header, coerce(items[index])])));
}

function coerce(value) {
  if (value === undefined || value === "") return "";
  const number = Number(value);
  return Number.isFinite(number) && value.trim() !== "" ? number : value;
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return parseCsv(await response.text());
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return value || "";
  if (Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(digits)}M`;
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(digits)}K`;
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

function intervalLabel(dataset) {
  const interval = String(dataset).split("_")[0].replace("m", "");
  return interval === "1440" ? "1D" : `${interval}m`;
}

function datasetType(dataset) {
  return String(dataset).includes("BREAKOUTS") ? "BREAKOUTS" : "ALL";
}

function modelFamily(model) {
  if (model.includes("FINBERT")) return "LSTM + FinBERT";
  if (model.includes("NEWS")) return "LSTM + News";
  if (model === "LSTM") return "LSTM OHLCV";
  return model;
}

function combineModels(data) {
  return [...data.modelResults, ...data.lstm, ...data.lstmNews, ...data.lstmFinbert].map((row) => ({
    ...row,
    interval: intervalLabel(row.dataset),
    dataset_type: datasetType(row.dataset),
    family: modelFamily(row.model)
  }));
}

function grouped(rows, key) {
  return rows.reduce((acc, row) => {
    const name = row[key];
    acc[name] ||= [];
    acc[name].push(row);
    return acc;
  }, {});
}

function renderKpis(data) {
  const allModels = combineModels(data);
  const best = allModels.reduce((winner, row) => row.roc_auc > winner.roc_auc ? row : winner, allModels[0]);
  const totalRows = data.stockSummary.reduce((sum, row) => sum + row.rows, 0);
  const articles = data.newsDaily.reduce((sum, row) => sum + row.article_count, 0);
  const aligned = data.finbert.length;
  const cards = [
    [formatNumber(totalRows, 1), "cleaned OHLCV rows across AAPL and AMZN"],
    [formatNumber(articles, 1), "daily-counted news articles"],
    [formatNumber(aligned, 1), "market bars with FinBERT-aligned news"],
    [best.roc_auc.toFixed(3), `${best.family} best ROC AUC on ${best.dataset}`]
  ];
  document.getElementById("kpis").innerHTML = cards.map(([value, label]) => `
    <div class="kpi"><strong>${value}</strong><span>${label}</span></div>
  `).join("");
}

function renderBarChart(target, rows, { labelKey, valueKey, colorKey, maxValue, formatter = formatNumber }) {
  const el = document.getElementById(target);
  const width = 760;
  const rowHeight = 34;
  const margin = { top: 14, right: 78, bottom: 28, left: 132 };
  const height = margin.top + margin.bottom + rows.length * rowHeight;
  const max = maxValue || Math.max(...rows.map((row) => row[valueKey]), 1);
  const barWidth = width - margin.left - margin.right;
  const items = rows.map((row, index) => {
    const y = margin.top + index * rowHeight;
    const w = Math.max(2, (row[valueKey] / max) * barWidth);
    const color = colors[Math.abs(hash(row[colorKey] || row[labelKey])) % colors.length];
    const delay = Math.min(index * 35, 420);
    return `
      <text class="tick" style="animation-delay:${delay}ms" x="${margin.left - 10}" y="${y + 21}" text-anchor="end">${row[labelKey]}</text>
      <rect class="animated-bar" style="animation-delay:${delay}ms" x="${margin.left}" y="${y + 7}" width="${w}" height="18" rx="4" fill="${color}"></rect>
      <text class="bar-label" style="animation-delay:${delay + 120}ms" x="${margin.left + w + 8}" y="${y + 21}">${formatter(row[valueKey])}</text>
    `;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img">${items}</svg>`;
}

function hash(value) {
  return String(value).split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
}

function renderLeaderboard() {
  const { metric, dataset, data } = state;
  const rows = combineModels(data).filter((row) => row.dataset_type === dataset);
  const bestByInterval = Object.entries(grouped(rows, "interval")).map(([interval, intervalRows]) => {
    const winner = intervalRows.reduce((best, row) => row[metric] > best[metric] ? row : best, intervalRows[0]);
    return {
      label: `${interval} ${winner.family}`,
      value: winner[metric],
      family: winner.family
    };
  }).sort((a, b) => b.value - a.value);

  document.getElementById("leaderboardLabel").textContent = `${metricLabels[metric]} on ${dataset.toLowerCase()}`;
  renderBarChart("leaderboardChart", bestByInterval, {
    labelKey: "label",
    valueKey: "value",
    colorKey: "family",
    maxValue: 1,
    formatter: (value) => value.toFixed(3)
  });
}

function renderFamilyChart() {
  const { metric, dataset, data } = state;
  const rows = combineModels(data).filter((row) => row.dataset_type === dataset);
  const familyRows = Object.entries(grouped(rows, "family")).map(([family, familyRows]) => ({
    label: family,
    value: familyRows.reduce((sum, row) => sum + row[metric], 0) / familyRows.length,
    family
  })).sort((a, b) => b.value - a.value);
  renderBarChart("familyChart", familyRows, {
    labelKey: "label",
    valueKey: "value",
    colorKey: "family",
    maxValue: 1,
    formatter: (value) => value.toFixed(3)
  });
}

function renderTopRuns() {
  const rows = combineModels(state.data)
    .filter((row) => row.dataset_type === state.dataset)
    .sort((a, b) => b[state.metric] - a[state.metric])
    .slice(0, 12);
  renderTable("topRunsTable", rows, [
    ["dataset", "Dataset"],
    ["family", "Model"],
    [state.metric, metricLabels[state.metric]],
    ["roc_auc", "ROC AUC"],
    ["f1", "F1"],
    ["accuracy", "Accuracy"]
  ]);
}

function renderTable(target, rows, columns) {
  const header = columns.map(([, label]) => `<th>${label}</th>`).join("");
  const body = rows.map((row) => `<tr>${columns.map(([key]) => {
    const value = row[key];
    const display = Number.isFinite(value) ? value.toFixed(3) : value;
    return `<td>${display}</td>`;
  }).join("")}</tr>`).join("");
  document.getElementById(target).innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderStockCharts(data) {
  const stockRows = data.stockSummary
    .map((row) => ({ label: `${row.ticker} ${intervalLabel(`${row.interval_minutes}m_ALL`)}`, value: row.rows, ticker: row.ticker }))
    .sort((a, b) => b.value - a.value);
  renderBarChart("stockRowsChart", stockRows, { labelKey: "label", valueKey: "value", colorKey: "ticker" });

  const volatilityRows = data.stockSummary
    .map((row) => ({ label: `${row.ticker} ${intervalLabel(`${row.interval_minutes}m_ALL`)}`, value: row.volatility_pct, ticker: row.ticker }))
    .sort((a, b) => b.value - a.value);
  renderBarChart("volatilityChart", volatilityRows, {
    labelKey: "label",
    valueKey: "value",
    colorKey: "ticker",
    formatter: (value) => `${value.toFixed(2)}%`
  });

  renderTable("splitsTable", data.splits, [
    ["ticker", "Ticker"],
    ["interval_minutes", "Interval"],
    ["train_rows", "Train Rows"],
    ["test_rows", "Test Rows"],
    ["train_end_utc", "Train End UTC"],
    ["test_start_utc", "Test Start UTC"]
  ]);
}

function renderLineChart(target, rows) {
  const el = document.getElementById(target);
  const width = 760;
  const height = 320;
  const margin = { top: 18, right: 24, bottom: 38, left: 54 };
  const xMax = rows.length - 1;
  const yMax = Math.max(...rows.map((row) => row.article_count), 1);
  const x = (index) => margin.left + (index / xMax) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - (value / yMax) * (height - margin.top - margin.bottom);
  const step = Math.max(1, Math.floor(rows.length / 220));
  const sampled = rows.filter((_, index) => index % step === 0 || index === rows.length - 1);
  const points = sampled.map((row, index) => {
    const originalIndex = Math.min(index * step, rows.length - 1);
    return `${x(originalIndex)},${y(row.article_count)}`;
  }).join(" ");
  const area = `${margin.left},${height - margin.bottom} ${points} ${width - margin.right},${height - margin.bottom}`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const row = rows[Math.floor(ratio * (rows.length - 1))];
    return `<text class="tick" x="${margin.left + ratio * (width - margin.left - margin.right)}" y="${height - 12}" text-anchor="middle">${row.published_date_utc.slice(0, 7)}</text>`;
  }).join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#dce3de"></line>
      <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#dce3de"></line>
      <polygon class="area-fill" points="${area}" fill="#d9e9e1"></polygon>
      <polyline class="line-path" pathLength="1" points="${points}" fill="none" stroke="#1f6d59" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"></polyline>
      <text class="tick" x="12" y="${y(yMax) + 4}">${formatNumber(yMax, 0)}</text>
      <text class="tick" x="12" y="${height - margin.bottom + 4}">0</text>
      ${ticks}
    </svg>
  `;
}

function renderSentiment(data) {
  const byTicker = Object.entries(grouped(data.finbert, "ticker")).map(([ticker, rows]) => {
    const weighted = rows.reduce((sum, row) => sum + row.finbert_sentiment * row.weighted_article_count, 0);
    const weights = rows.reduce((sum, row) => sum + row.weighted_article_count, 0);
    return { label: ticker, value: weighted / weights, ticker };
  });
  renderDivergingChart("sentimentChart", byTicker);
}

function renderDivergingChart(target, rows) {
  const el = document.getElementById(target);
  const width = 760;
  const rowHeight = 56;
  const margin = { top: 38, right: 130, bottom: 42, left: 118 };
  const height = margin.top + margin.bottom + rows.length * rowHeight;
  const plotWidth = width - margin.left - margin.right;
  const zeroX = margin.left + plotWidth / 2;
  const maxAbs = Math.max(...rows.map((row) => Math.abs(row.value)), 0.01);
  const scale = (plotWidth / 2 - 18) / maxAbs;
  const axisTicks = [-maxAbs, 0, maxAbs].map((value) => {
    const x = zeroX + value * scale;
    const label = value === 0 ? "0" : value.toFixed(2);
    return `
      <line x1="${x}" y1="${margin.top - 12}" x2="${x}" y2="${height - margin.bottom + 8}" stroke="${value === 0 ? "#789087" : "#d9e3df"}" stroke-width="${value === 0 ? 2 : 1}"></line>
      <text class="tick" x="${x}" y="${height - 14}" text-anchor="middle">${label}</text>
    `;
  }).join("");
  const items = rows.map((row, index) => {
    const y = margin.top + index * rowHeight + 12;
    const barWidth = Math.max(4, Math.abs(row.value) * scale);
    const x = row.value < 0 ? zeroX - barWidth : zeroX;
    const color = row.value < 0 ? "#d56f55" : "#1f6d59";
    const delay = index * 70;
    return `
      <text class="tick" style="animation-delay:${delay}ms" x="${margin.left - 16}" y="${y + 20}" text-anchor="end">${row.label}</text>
      <rect class="animated-bar" style="animation-delay:${delay}ms" x="${x}" y="${y + 4}" width="${barWidth}" height="22" rx="6" fill="${color}"></rect>
      <text class="bar-label" style="animation-delay:${delay + 120}ms" x="${width - 28}" y="${y + 21}" text-anchor="end">${row.value.toFixed(3)}</text>
    `;
  }).join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Weighted FinBERT sentiment by ticker">
      ${axisTicks}
      <text class="tick" x="${margin.left}" y="22">Negative</text>
      <text class="tick" x="${width - margin.right}" y="22" text-anchor="end">Positive</text>
      <text class="tick" x="${width - 28}" y="22" text-anchor="end">Score</text>
      ${items}
    </svg>
  `;
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let table = [];
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((line) => !/^\|?\s*:?-+:?\s*\|/.test(line));
    const cells = rows.map((line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
    const [head, ...body] = cells;
    html += `<div class="table-wrap"><table><thead><tr>${head.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    table = [];
  };
  for (const line of lines) {
    if (line.trim().startsWith("|")) {
      table.push(line);
      continue;
    }
    flushTable();
    if (line.startsWith("# ")) html += `<h1>${line.slice(2)}</h1>`;
    else if (line.startsWith("## ")) html += `<h2>${line.slice(3)}</h2>`;
    else if (line.startsWith("### ")) html += `<h3>${line.slice(4)}</h3>`;
    else if (line.startsWith("- ")) html += `<p>${line.slice(2)}</p>`;
    else if (line.trim()) html += `<p>${line}</p>`;
  }
  flushTable();
  return html
    .replaceAll("`EVALUATIONS/analysis_outputs", "<code>EVALUATIONS/analysis_outputs")
    .replaceAll("`analysis_outputs", "<code>EVALUATIONS/analysis_outputs")
    .replaceAll("/`", "/</code>");
}

function removeMarkdownSection(markdown, heading) {
  const pattern = new RegExp(`^## ${heading}\\s*$[\\s\\S]*?(?=^##\\s|$)`, "im");
  return markdown.replace(pattern, "").trimStart();
}

async function setupReports() {
  const tabs = document.getElementById("reportTabs");
  const content = document.getElementById("reportContent");
  async function selectReport(index) {
    [...tabs.children].forEach((button, buttonIndex) => button.classList.toggle("active", buttonIndex === index));
    content.innerHTML = `<div class="loading">Loading report...</div>`;
    const response = await fetch(reportPaths[index][1]);
    let markdown = await response.text();
    if (reportPaths[index][0] === "EDA") {
      markdown = removeMarkdownSection(markdown, "Proposal Alignment");
    }
    content.innerHTML = markdownToHtml(markdown);
  }
  tabs.innerHTML = reportPaths.map(([name], index) => `<button type="button" data-index="${index}">${name}</button>`).join("");
  tabs.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (button) selectReport(Number(button.dataset.index));
  });
  await selectReport(0);
}

function wireControls() {
  document.getElementById("metricSelect").addEventListener("change", (event) => {
    state.metric = event.target.value;
    renderModelSections();
  });
  document.getElementById("datasetSelect").addEventListener("change", (event) => {
    state.dataset = event.target.value;
    renderModelSections();
  });
}

function renderModelSections() {
  renderLeaderboard();
  renderFamilyChart();
  renderTopRuns();
}

async function init() {
  try {
    document.querySelectorAll(".chart").forEach((chart) => chart.innerHTML = `<div class="loading">Loading data...</div>`);
    const [modelResults, lstm, lstmNews, lstmFinbert, stockSummary, splits, newsDaily, finbert] = await Promise.all([
      loadCsv(paths.modelResults),
      loadCsv(paths.lstm),
      loadCsv(paths.lstmNews),
      loadCsv(paths.lstmFinbert),
      loadCsv(paths.stockSummary),
      loadCsv(paths.splits),
      loadCsv(paths.newsDaily),
      loadCsv(paths.finbert)
    ]);
    state.data = { modelResults, lstm, lstmNews, lstmFinbert, stockSummary, splits, newsDaily, finbert };
    renderKpis(state.data);
    renderModelSections();
    renderStockCharts(state.data);
    renderLineChart("newsChart", state.data.newsDaily);
    renderSentiment(state.data);
    await setupReports();
    wireControls();
  } catch (error) {
    document.body.insertAdjacentHTML("afterbegin", `<div class="error">Dashboard could not load all files. Start a local web server from the repo root and open <code>/CODE/dashboard/</code>. ${error.message}</div>`);
  }
}

init();
