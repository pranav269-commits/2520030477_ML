(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const state = { summary: null, hyper: [] };

  const samples = {
    spam: "URGENT: Congratulations! You have won a free cash reward. Click the link now and claim your prize before the offer expires today.",
    ham: "Hi, the machine learning project meeting has been moved to 3 PM tomorrow. Please bring the updated report and your notes."
  };

  const pct = (value, digits = 2) => value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
  const niceKernel = (name) => ({ linear: "Linear", rbf: "RBF", poly: "Polynomial", sigmoid: "Sigmoid" }[name] || name || "—");

  function toast(message) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => node.classList.remove("show"), 3600);
  }

  function setStatus(label, ready = false) {
    const status = $("#systemStatus");
    status.classList.toggle("ready", ready);
    status.querySelector("span:last-child").textContent = label;
  }

  async function getJSON(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function observeReveals() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: .12 });
    $$(".reveal").forEach(el => observer.observe(el));
  }

  function renderDataset(summary) {
    const d = summary.dataset;
    $("#heroRows").textContent = Number(d.raw_rows || d.clean_rows || 5572).toLocaleString();
    const values = [d.clean_rows, d.ham, d.spam, d.features, d.test_rows];
    $$("#datasetStats strong").forEach((node, i) => node.textContent = Number(values[i] || 0).toLocaleString());
    $("#baselineWinner").textContent = `Baseline F1 winner · ${summary.experiment.baseline_winner}`;
  }

  function renderModelMetrics(summary) {
    const metrics = [
      ["Accuracy", "accuracy", true], ["Precision", "precision", true], ["Recall", "recall", true],
      ["F1-Score", "f1", true], ["Specificity", "specificity", true], ["ROC-AUC", "roc_auc", true],
      ["False Positive Rate", "fpr", false], ["False Negative Rate", "fnr", false]
    ];
    const m = summary.model_metrics;
    const tbody = $("#modelMetricTable tbody");
    tbody.innerHTML = metrics.map(([label, key, higher]) => {
      const vals = [m.naive_bayes[key], m.svm_baseline[key], m.svm_optimized[key]].map(v => v == null ? NaN : Number(v));
      const target = higher ? Math.max(...vals.filter(Number.isFinite)) : Math.min(...vals.filter(Number.isFinite));
      const cell = (v) => `<td class="${Number(v) === target ? "best-cell" : ""}">${pct(v)}</td>`;
      return `<tr><td>${label}</td>${cell(vals[0])}${cell(vals[1])}${cell(vals[2])}</tr>`;
    }).join("");

    const barMetrics = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1", "f1"]];
    $("#metricBars").innerHTML = barMetrics.map(([label, key]) => `
      <div class="metric-group">
        <span><b>${label}</b><i>NB / OPT-SVM</i></span>
        <div class="comparison-bar"><i style="width:${m.naive_bayes[key] * 100}%"></i><b><span>NB</span><span>${pct(m.naive_bayes[key])}</span></b></div>
        <div class="comparison-bar svm"><i style="width:${m.svm_optimized[key] * 100}%"></i><b><span>SVM</span><span>${pct(m.svm_optimized[key])}</span></b></div>
      </div>`).join("");
  }

  function renderMatrix(target, matrix) {
    const [row0, row1] = matrix;
    target.innerHTML = `
      <div></div><div class="matrix-label">Pred. Ham</div><div class="matrix-label">Pred. Spam</div>
      <div class="matrix-label">Actual Ham</div><div class="matrix-cell">${row0[0]}</div><div class="matrix-cell error">${row0[1]}</div>
      <div class="matrix-label">Actual Spam</div><div class="matrix-cell error">${row1[0]}</div><div class="matrix-cell">${row1[1]}</div>`;
  }

  function renderKernels(summary) {
    const rows = summary.kernel_results;
    const best = [...rows].sort((a, b) => b.f1 - a.f1)[0];
    $("#kernelCards").innerHTML = rows.map(row => `
      <article class="kernel-card ${row.kernel === best.kernel ? "best" : ""}">
        <span>${row.kernel === best.kernel ? "BEST BASELINE KERNEL" : "SVM KERNEL"}</span>
        <h3>${niceKernel(row.kernel)}</h3>
        <div class="kernel-score">${pct(row.f1)}</div>
        <small>F1 score</small>
      </article>`).join("");

    $("#kernelTable tbody").innerHTML = rows.map(row => `
      <tr class="${row.kernel === best.kernel ? "best-row" : ""}">
        <td>${niceKernel(row.kernel)}</td><td>${pct(row.accuracy)}</td><td>${pct(row.precision)}</td>
        <td>${pct(row.recall)}</td><td>${pct(row.f1)}</td><td>${pct(row.roc_auc)}</td>
      </tr>`).join("");
  }

  function paramsLine(row) {
    const bits = [`C = ${row.C}`];
    if (row.gamma != null) bits.push(`gamma = ${row.gamma}`);
    if (row.degree != null) bits.push(`degree = ${row.degree}`);
    return bits.join(" · ");
  }

  function renderHyper(rows) {
    const filter = $("#kernelFilter").value;
    const visible = (filter === "all" ? rows : rows.filter(row => row.kernel === filter)).slice(0, 30);
    $("#hyperTable tbody").innerHTML = visible.map(row => `
      <tr class="${row.rank === 1 ? "best-row" : ""}">
        <td>#${String(row.rank).padStart(2, "0")}</td><td>${niceKernel(row.kernel)}</td><td>${row.C}</td>
        <td>${row.gamma ?? "—"}</td><td>${row.degree ?? "—"}</td><td>${pct(row.accuracy)}</td>
        <td>${pct(row.precision)}</td><td>${pct(row.recall)}</td><td>${pct(row.f1)}</td><td>${pct(row.specificity)}</td>
      </tr>`).join("");
  }

  function renderBest(summary) {
    const best = summary.hyperparameter_results[0];
    $("#bestKernel").textContent = `${niceKernel(best.kernel)} SVM`;
    $("#bestParamsLine").textContent = paramsLine(best);
    $("#bestF1").textContent = pct(best.f1);
    $("#bestAccuracy").textContent = pct(best.accuracy);
  }

  function renderSummary(summary) {
    state.summary = summary;
    state.hyper = summary.hyperparameter_results || [];
    renderDataset(summary);
    renderModelMetrics(summary);
    renderMatrix($("#nbMatrix"), summary.model_metrics.naive_bayes.confusion_matrix);
    renderMatrix($("#svmMatrix"), summary.model_metrics.svm_optimized.confusion_matrix);
    renderKernels(summary);
    renderBest(summary);
    renderHyper(state.hyper);
  }

  async function loadSummary() {
    setStatus("Preparing model", false);
    try {
      const summary = await getJSON("/api/summary");
      renderSummary(summary);
      setStatus("Model ready", true);
    } catch (error) {
      setStatus("Setup required", false);
      toast(error.message);
    }
  }

  function renderEvidence(tokens) {
    const box = $("#evidenceList");
    if (!tokens.length) {
      box.innerHTML = `<p class="table-note">No submitted terms matched the trained TF-IDF vocabulary strongly enough for a token-level explanation.</p>`;
      return;
    }
    box.innerHTML = tokens.map(item => `
      <div class="evidence-row ${item.direction}">
        <strong>${escapeHTML(item.token)}</strong>
        <div class="evidence-bar"><i style="width:${Math.max(5, item.strength * 100)}%"></i></div>
        <span>${item.direction} · ${Math.round(item.strength * 100)}</span>
      </div>`).join("");
  }

  function renderProfile(profile) {
    const order = [["urgency","Urgency"],["reward","Reward"],["action","Action"],["money","Money"],["promotion","Promotion"],["normality","Normality"]];
    $("#profileBars").innerHTML = order.map(([key, label]) => `
      <div class="profile-item"><label>${label}</label><div class="profile-track"><i style="width:${profile[key]}%"></i></div><span>${profile[key]}</span></div>`).join("");
  }

  function renderAnalysis(result) {
    $("#resultEmpty").hidden = true;
    $("#resultContent").hidden = false;
    $("#finalVerdict").textContent = `${result.final_verdict} detected`;
    $("#agreementLabel").textContent = result.agreement ? "Models agree" : "Model disagreement";
    $("#agreementCode").textContent = result.agreement ? "NB = SVM" : "NB ≠ SVM";

    $("#nbPrediction").textContent = result.naive_bayes.prediction;
    $("#nbConfidence").textContent = pct(result.naive_bayes.confidence);
    $("#nbSpam").textContent = pct(result.naive_bayes.spam_probability);
    $("#nbHam").textContent = pct(result.naive_bayes.ham_probability);
    $("#nbTrack").style.width = `${result.naive_bayes.spam_probability * 100}%`;

    $("#svmPrediction").textContent = result.svm.prediction;
    $("#svmStrength").textContent = pct(result.svm.decision_strength);
    $("#svmStrengthDetail").textContent = pct(result.svm.decision_strength);
    $("#svmMargin").textContent = Number(result.svm.decision_margin).toFixed(4);
    $("#svmKernel").textContent = niceKernel(result.svm.kernel);
    const p = [`C ${result.svm.C}`];
    if (result.svm.gamma != null) p.push(`γ ${result.svm.gamma}`);
    if (result.svm.degree != null) p.push(`degree ${result.svm.degree}`);
    $("#svmParams").textContent = p.join(" · ");
    $("#svmTrack").style.width = `${result.svm.boundary_position * 100}%`;

    renderEvidence(result.evidence_tokens || []);
    renderProfile(result.message_profile);
    const s = result.message_stats;
    $("#messageStats").textContent = `${s.words} words · ${s.characters} chars · ${s.digits} digits`;
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function analyze() {
    const input = $("#emailInput");
    const text = input.value.trim();
    if (!text) return toast("Enter a message before running the analysis.");
    const button = $("#analyzeBtn");
    button.classList.add("loading");
    button.disabled = true;
    button.querySelector(".button-label").textContent = "Analyzing with NB + SVM…";
    try {
      const result = await getJSON("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text })
      });
      renderAnalysis(result);
    } catch (error) {
      toast(error.message);
    } finally {
      button.classList.remove("loading");
      button.disabled = false;
      button.querySelector(".button-label").textContent = "Run dual-model analysis";
    }
  }

  function escapeHTML(value) {
    return String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  }

  function bindUI() {
    const input = $("#emailInput");
    input.addEventListener("input", () => $("#charCount").textContent = `${input.value.length.toLocaleString()} / 15,000`);
    $$(".sample-btn").forEach(btn => btn.addEventListener("click", () => {
      input.value = samples[btn.dataset.sample];
      input.dispatchEvent(new Event("input"));
      input.focus();
    }));
    $("#clearBtn").addEventListener("click", () => { input.value = ""; input.dispatchEvent(new Event("input")); input.focus(); });
    $("#analyzeBtn").addEventListener("click", analyze);
    $("#kernelFilter").addEventListener("change", () => renderHyper(state.hyper));
    $("#kernelVizToggle").addEventListener("click", () => {
      const viz = $("#kernelViz");
      const transformed = viz.classList.toggle("transformed");
      $("#kernelVizState").textContent = transformed ? "Kernel-transformed view" : "Original feature space";
      $("#kernelVizCopy").textContent = transformed
        ? "The similarity function lets SVM behave as if samples were mapped to a space where a clean separating hyperplane can exist."
        : "Class patterns can overlap in their original representation, making a straight separator difficult.";
      $("#kernelVizToggle").textContent = transformed ? "Reset space" : "Transform space";
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    observeReveals();
    bindUI();
    loadSummary();
  });
})();
