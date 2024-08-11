function createChart(ctx, data, label = null) {
    let maxVal = Math.ceil(Math.max(...Object.values(data)) * 1.15);
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)', 'rgba(75, 192, 192, 0.2)', 'rgba(153, 102, 255, 0.2)'],
                borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)'],
                borderWidth: 3,
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                    },
                    afterDataLimits: function (scale) {
                        scale.max = maxVal;
                    }

                }
            }
        }
    });
}

function updateChart(projects_pair, text) {
    let ctx = document.getElementById('myChart').getContext('2d');
    let [firstPr, secPr] = projects_pair.split('-');

    let dataPaths = {
        'different label': data.difference_between_projects[projects_pair][text]['differ_label_count'],
        'different coords': data.difference_between_projects[projects_pair][text]['differ_coords_count'],
        'full match': data.difference_between_projects[projects_pair][text]['full_match_count'],
        [`total project ${firstPr}`]: data.amount_anns_per_project[firstPr][text]['total'],
        [`total project ${secPr}`]: data.amount_anns_per_project[secPr][text]['total']
    };

    if (chart) chart.destroy();
    createChart(ctx, dataPaths, text);
}

function updateTextOptions() {
    var projectPair = projectSelect.value;
    textSelect.innerHTML = '';
    for (var text in data.difference_between_projects[projectPair]) {
        var option = document.createElement('option');
        option.value = text;
        option.text = text.slice(0, 50);
        textSelect.appendChild(option);
    }
    updateChart(projectPair, textSelect.value);
}

let projectSelect = document.createElement('select');
projectSelect.id = 'projectSelect';

for (var projectPair in data.difference_between_projects) {
    let option = document.createElement('option');
    option.value = projectPair;
    option.text = projectPair;
    projectSelect.appendChild(option);
}

document.body.appendChild(projectSelect);

let textSelect = document.createElement('select');
textSelect.id = 'textSelect';
document.body.appendChild(textSelect);

let chart;

document.getElementById('projectSelect').addEventListener('change', updateTextOptions);
document.getElementById('textSelect').addEventListener('change', function () {
    updateChart(projectPair, this.value);
});

updateTextOptions();