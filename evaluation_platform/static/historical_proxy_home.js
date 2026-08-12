/* Historical proxy display for the two existing homepage metric views. */
(() => {
  let metric = new URLSearchParams(window.location.search).get('metric') || '';
  const button = document.getElementById('loadHistoricalProxyBtn');
  if (!button) return;

  const supports = value => value === 'command_execution_acceptance' || value === 'dynamic_separation_adjustment';
  button.style.display = supports(metric) ? '' : 'none';

  const setText = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  };
  const setHtml = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.innerHTML = value;
  };
  const pct = value => value == null ? '不适用' : (Number(value) * 100).toFixed(2);
  const count = value => value == null ? '—' : Number(value).toLocaleString('zh-CN');
  const endpoint = (window.location.protocol === 'file:' ? 'http://127.0.0.1:8765' : '') + '/api/historical-atc-proxy';

  function showAcceptance(data) {
    const p = data.historical_instruction_operational_plausibility || {};
    setText('macroLabel', '历史指令可执行性代理');
    setText('macroValue', pct(p.rate)); setText('macroUnit', '%');
    setText('macroNote', '历史进近数据：指令后 20–60 秒的可观测响应代理');
    setText('precisionLabel', '可观测历史指令'); setText('precisionValue', count(p.observable_denominator));
    setText('strictLabel', '观察到方向一致响应'); setText('strictValue', count(p.accepted_observed_response));
    setText('inflationLabel', '条件不足/未通过'); setText('inflationValue', count((p.conditional || 0) + (p.rejected || 0)));
    setText('formulaNote', '历史数据代理结果：不含飞行员复诵或人工管制员评分，因此不等同于真实管制指令接受度。');
    setHtml('resultBoundary', '<b>结果适用边界</b><br>本结果仅表示历史指令后的可观测轨迹响应，不能解释为飞行员复诵接受率或人工管制员主观接受度。');
    setHtml('matchRows', [
      ['观察到方向一致响应', count(p.accepted_observed_response), '代理接受'],
      ['未通过', count(p.rejected), '无状态关联、无效指令或未观察到响应'],
      ['条件不足', count(p.conditional), '缺少后续轨迹或无法判定方向'],
    ].map(row => `<tr><td colspan="2">${row[0]}</td><td colspan="2">${row[1]}</td><td colspan="3">${row[2]}</td></tr>`).join(''));
  }

  function showSeparation(data) {
    const s = data.post_instruction_separation_maintenance || {};
    /* Deliberately do not change macroValue: that card is the formal metric. */
    setText('precisionLabel', '历史近距交通样本'); setText('precisionValue', count(s.nearby_instruction_count));
    setText('strictLabel', '历史间隔保持样本'); setText('strictValue', count(s.nearby_maintained));
    setText('inflationLabel', '初始间隔突破样本'); setText('inflationValue', count(s.initial_loss_of_separation_count));
    setText('formulaNote', `历史辅助分析：近距交通间隔保持率 ${pct(s.nearby_traffic_maintenance_rate)}%。它不覆盖正式动态间隔调整成功率。`);
    setHtml('resultBoundary', '<b>结果适用边界</b><br>已载入历史进近间隔保持分析。该数据没有用户修改间隔标准的事件，不能替代正式动态间隔调整成功率。');
    setHtml('matchRows', [
      ['近距交通间隔保持率（历史辅助）', pct(s.nearby_traffic_maintenance_rate) + '%', '后续 60 秒满足 3 NM 或 1000 ft'],
      ['附近交通样本', count(s.nearby_instruction_count), '指令时刻 5 NM 内有其他航空器'],
      ['初始间隔突破样本', count(s.initial_loss_of_separation_count), '为 0，不能计算历史恢复率'],
    ].map(row => `<tr><td colspan="2">${row[0]}</td><td colspan="2">${row[1]}</td><td colspan="3">${row[2]}</td></tr>`).join(''));
  }

  button.addEventListener('click', async () => {
    if (!supports(metric)) return;
    const status = document.getElementById('runStatus');
    const original = button.textContent;
    button.disabled = true;
    button.textContent = '正在载入...';
    if (status) status.textContent = '● 正在读取历史进近数据代理结果';
    try {
      const response = await fetch(endpoint, {cache: 'no-store'});
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || `HTTP ${response.status}`);
      if (metric === 'command_execution_acceptance') showAcceptance(data);
      else showSeparation(data);
      if (status) status.textContent = '● 历史数据代理结果已载入';
      const run = document.getElementById('runId');
      if (run) run.textContent = 'HISTORICAL ATC PROXY';
      button.textContent = '已载入历史数据';
    } catch (error) {
      if (status) status.textContent = `● 历史数据载入失败：${error.message}`;
      button.textContent = '载入失败，点击重试';
      button.title = error.message;
    } finally {
      button.disabled = false;
      if (button.textContent === original) button.textContent = original;
    }
  });

  window.updateHistoricalProxyMetric = nextMetric => {
    metric = nextMetric || '';
    button.style.display = supports(metric) ? '' : 'none';
    button.disabled = false;
    button.textContent = '载入历史数据代理结果';
    button.removeAttribute('title');
  };
})();
