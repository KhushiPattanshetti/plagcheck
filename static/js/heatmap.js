function initHeatmap(data) {
    // Validate data
    if (!data || !data.scores || data.scores.length === 0) {
        document.getElementById('heatmap').innerHTML = 
            '<div class="heatmap-error">No valid data available for visualization</div>';
        return;
    }

    // Determine dataset size characteristics
    const rows = data.scores.length;
    const cols = data.scores[0].length;
    const isSmallDataset = rows <= 15 && cols <= 15;
    const totalCells = rows * cols;

    // Dynamic configuration
    const config = {
        cellSize: isSmallDataset ? 50 : 20,
        fontSize: isSmallDataset ? 12 : 8,
        showValues: isSmallDataset || totalCells < 100,
        showAxes: isSmallDataset,
        showHover: isSmallDataset,
        minHeight: 300
    };

    // Color scale
    const colorScale = [
        [0.0, '#FFFFFF'], [0.1, '#FFEEEE'],
        [0.3, '#FFCCCC'], [0.5, '#FF9999'],
        [0.7, '#FF6666'], [0.9, '#CC0000'],
        [1.0, '#990000']
    ];

    // Calculate dimensions - now using 100% of container width
    const container = document.getElementById('heatmap').parentElement;
    const containerWidth = container.clientWidth;
    const height = Math.max(config.minHeight, rows * config.cellSize);

    // Prepare text display
    const textMatrix = config.showValues ? 
        data.scores.map(row => row.map(v => v.toFixed(2))) : 
        null;

    // Create heatmap trace
    const heatmapTrace = {
        z: data.scores,
        x: Array.from({length: cols}, (_, i) => i + 1),
        y: config.showAxes ? Array.from({length: rows}, (_, i) => `Row ${i+1}`) : [],
        type: 'heatmap',
        colorscale: colorScale,
        zmin: 0,
        zmax: 1,
        hoverinfo: config.showHover ? 'x+y+z' : 'none',
        showscale: true,
        text: textMatrix,
        texttemplate: config.showValues ? "%{text}" : "",
        textfont: {
            size: config.fontSize,
            color: config.showValues ? 
                data.scores.map(row => row.map(v => v > 0.5 ? 'white' : '#333')) : []
        },
        colorbar: {
            title: 'Probability',
            titleside: 'right',
            thickness: 15,
            len: 0.75,
            tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext: ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
        }
    };

    // Layout configuration for full width
    const layout = {
        title: {
            text: `N-gram Probability Heatmap (n=${data.stats.n_value})`,
            font: {size: 18}
        },
        width: containerWidth,
        height: height,
        margin: {
            l: config.showAxes ? 80 : 50,
            r: 50,
            b: config.showAxes ? 60 : 50,
            t: 80,
            pad: 10
        },
        xaxis: {
            visible: config.showAxes,
            title: config.showAxes ? 'Position' : '',
            tickfont: {size: config.fontSize},
            showgrid: false
        },
        yaxis: {
            visible: config.showAxes,
            autorange: 'reversed',
            tickfont: {size: config.fontSize},
            showgrid: false
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        autosize: true
    };

    // Plot configuration
    const plotConfig = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
        displaylogo: false
    };

    // Create the plot
    const heatmapDiv = document.getElementById('heatmap');
    Plotly.purge(heatmapDiv);
    Plotly.newPlot(heatmapDiv, [heatmapTrace], layout, plotConfig);

    // Add responsive behavior
    window.addEventListener('resize', function() {
        const newWidth = heatmapDiv.parentElement.clientWidth;
        Plotly.relayout(heatmapDiv, {
            width: newWidth,
            'xaxis.autorange': true,
            'yaxis.autorange': true
        });
    });
}

// Optional: Make truly full screen
function enterFullscreen() {
    const heatmapDiv = document.getElementById('heatmap');
    if (heatmapDiv.requestFullscreen) {
        heatmapDiv.requestFullscreen();
    }
    // Add similar for other browsers' fullscreen APIs
}