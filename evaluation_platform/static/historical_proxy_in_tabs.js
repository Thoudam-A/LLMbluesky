/* Shows historical ATC proxy results inside the existing H-PPO metric tabs. */
document.addEventListener('DOMContentLoaded', () => {
  const supported = new Set(['dynamic_separation_adjustment', 'command_acceptability_proxy']);
  const runButton = document.getElementById('runBtn');
  if (!runButton || !runButton.parentElement) return;

  let button = document.getElementById('historicalProxyBtn');
  if (!button) {
    button = document.createElement('button');
    button.id = 'historicalProxyBtn';
    button.type = 'button';
    button.textContent = '载入历史数据代理结果';
    runButton.parentElement.appendChild(button);
  }

  const text = (id, value) => { document.getElementById(id).textContent = value; };
  const percent = value => value == null ? '不适用' : (Number(value) * 100).toFixed(2) + '%';
  const integer = value => value == null ? '--' : Number(value).toLocaleString('zh-CN');
  const root = location.protocol === 'file:' ? 'http://127.0.0.1:8765' : '';

  function setCards(labels, values, notes) {
    ['secondary', 'third', 'fourth'].forEach((key, index) => {
      text(key + 'Label', labels[index]);
      text(key + 'Value', values[index]);
      text(key + 'Note', notes[index]);
    });
  }

  function render(payload) {
    const plausibility = payload.historical_instruction_operational_plausibility || {};
    const separation = payload.post_instruction_separation_maintenance || {};
    const isAcceptance = metric === 'command_acceptability_proxy';
    text('primaryValue', percent(isAcceptance ? plausibility.rate : separation.initial_loss_recovery_rate));
    text('primaryNote', isAcceptance
      ? '历史进近数据代理：指令后是否观察到方向一致的轨迹响应'
      : '历史进近数据代理：仅统计指令时刻已发生间隔突破的恢复样本');
    text('status', '已载入历史进近数据代理结果；不会覆盖或修改 H-PPO 运行日志评估。');
    document.getElementById('summary').innerHTML = '<div><span>数据口径</span><b>历史数据代理</b></div>';

    if (isAcceptance) {
      setCards(
        ['可观测指令', '观察到响应', '条件不足/未通过'],
        [integer(plausibility.observable_denominator), integer(plausibility.accepted_observed_response), integer((plausibility.conditional || 0) + (plausibility.rejected || 0))],
        ['存在前后状态、可判断响应', '20–60 秒内方向一致', '缺少后续轨迹或未观察到响应']
      );
      text('detailTitle', '历史指令可接受性代理：分类汇总');
      document.getElementById('tableHead').innerHTML = '<tr><th>结果类别</th><th>数量</th><th>说明</th></tr>';
      document.getElementById('detailRows').innerHTML = [
        ['观察到方向一致响应', integer(plausibility.accepted_observed_response), '代理接受'],
        ['未通过', integer(plausibility.rejected), '无状态关联、无效指令或未观察到响应'],
        ['条件不足', integer(plausibility.conditional), '缺少足够后续轨迹或无法判定方向'],
      ].map(row => `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td></tr>`).join('');
      text('boundary', '历史数据不含飞行员复诵或人工管制员评分。本页显示的是“历史指令后可观测轨迹响应”的自动化离线代理，不能直接等同于真实管制指令接受度。');
    } else {
      setCards(
        ['附近交通样本', '保持间隔样本', '初始间隔突破样本'],
        [integer(separation.nearby_instruction_count), integer(separation.nearby_maintained), integer(separation.initial_loss_of_separation_count)],
        ['指令时刻 5 NM 内有其他航空器', '后续窗口满足 3 NM 或 1000 ft', '本轮历史样本为 0，恢复率不适用']
      );
      text('detailTitle', '历史指令后间隔保持代理：汇总');
      document.getElementById('tableHead').innerHTML = '<tr><th>统计项</th><th>数量/结果</th><th>说明</th></tr>';
      document.getElementById('detailRows').innerHTML = [
        ['附近交通间隔保持率', percent(separation.nearby_traffic_maintenance_rate), '不是动态间隔调整成功率'],
        ['初始间隔突破后恢复率', percent(separation.initial_loss_recovery_rate), '只有发生初始突破时才可计算'],
        ['初始间隔突破样本', integer(separation.initial_loss_of_separation_count), '当前历史样本为 0'],
      ].map(row => `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td></tr>`).join('');
      text('boundary', '历史数据没有间隔标准变更事件，因此不能计算正式动态间隔调整成功率。本页仅展示历史指令后的间隔保持/恢复代理；正式指标仍以 BlueSky 受控间隔变更实验为准。');
    }
  }

  button.addEventListener('click', async () => {
    if (!supported.has(metric)) {
      toast('历史数据代理仅适用于动态间隔调整成功率和自动化离线管制指令可接受性。');
      return;
    }
    button.disabled = true;
    text('status', '正在载入历史进近数据代理结果...');
    try {
      const response = await fetch(root + '/api/historical-atc-proxy');
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message || '历史数据代理结果不可用');
      render(payload);
    } catch (error) {
      text('status', '无法载入历史数据代理结果：' + error.message);
    } finally {
      button.disabled = false;
    }
  });
});
