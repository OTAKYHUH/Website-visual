// static/js/daily.js
const raw = document.getElementById('daily-data').textContent;
const initialPayload = JSON.parse(raw);
window.addEventListener('DOMContentLoaded', () => {
  if (window.loadDeepDive) window.loadDeepDive(initialPayload);
});
// right after datasets are created inside makeOverviewAnimated
var lastIdx = labels.length - 1;
datasets.forEach(function(ds){
  ds.pointRadius = function(ctx){
    return (ctx.dataIndex === lastIdx) ? 6 : (mode === 'scatter' ? 3 : 0);
  };
  ds.pointHoverRadius = function(ctx){
    return (ctx.dataIndex === lastIdx) ? 8 : 6;
  };
});
const USE_ANIMATED_OVERVIEW = true;
let chOverview;
if (USE_ANIMATED_OVERVIEW) {
  chOverview = makeOverviewAnimated('overview', payload.hours, payload.overview, { mode:'line' });
} else {
  chOverview = makeChart('overview', payload.overview);
}

