function initHeatmap(data) {
  // Validate data
  if (!data || !data.scores || data.scores.length === 0) {
      console.error("Invalid heatmap data:", data);
      document.getElementById('heatmap').innerHTML = 
          '<div class="heatmap-error">No valid data available for visualization</div>';
      return;
  }

  // Enhanced color scale with more gradation
  const colorScale = [
      [0.0, '#FFFFFF'], // White
      [0.1, '#FFEEEE'], // Very light red
      [0.3, '#FFCCCC'], // Light red
      [0.5, '#FF9999'], // Medium light red
      [0.7, '#FF6666'], // Medium red
      [0.9, '#CC0000'], // Dark red
      [1.0, '#990000']  // Very dark red
  ];

  // Calculate dynamic dimensions
  const containerWidth = Math.min(1200, window.innerWidth - 40);
  const cellHeight = 25;
  const maxHeight = Math.min(800, data.scores.length * cellHeight);

  // Create heatmap trace
  const heatmapTrace = {
      z: data.scores,
      x: Array.from({length: data.scores[0].length}, (_, i) => i + 1),
      type: 'heatmap',
      colorscale: colorScale,
      zmin: 0,
      zmax: 1,
      hoverinfo: 'none',
      showscale: true,
      colorbar: {
          title: 'Probability',
          titleside: 'right',
          thickness: 15,
          len: 0.8,
          tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
          ticktext: ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
      }
  };

  // Layout configuration
  const layout = {
      title: {
          text: `N-gram Probability Distribution (n=${data.stats.n_value})`,
          font: {
              size: 18,
              family: 'Arial'
          }
      },
      width: containerWidth,
      height: maxHeight,
      margin: {
          l: 50,
          r: 50,
          b: 50,
          t: 80,
          pad: 10
      },
      xaxis: {
          visible: false,
          showgrid: false,
          zeroline: false
      },
      yaxis: {
          visible: false,
          showgrid: false,
          zeroline: false
      },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      hovermode: false
  };

  // Configuration options
  const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
      displaylogo: false
  };

  // Create or update the plot
  const heatmapDiv = document.getElementById('heatmap');
  Plotly.purge(heatmapDiv);
  Plotly.newPlot(heatmapDiv, [heatmapTrace], layout, config);

  // Add resize handler
  window.addEventListener('resize', function() {
      const update = {
          width: Math.min(1200, window.innerWidth - 40)
      };
      Plotly.relayout(heatmapDiv, update);
  });
}

// Error display styling (add to your CSS)
/*
.heatmap-error {
  padding: 2rem;
  text-align: center;
  color: #990000;
  font-weight: bold;
  background-color: #ffeeee;
  border-radius: 8px;
  margin: 1rem;
}
*/