// Dashboard charts and stats
const chartColors = {
    red: '#dc3545',
    green: '#28a745',
    yellow: '#ffc107',
    blue: '#007bff',
    gray: '#6c757d',
    lightRed: '#f8d7da',
    lightGreen: '#d4edda',
    lightYellow: '#fff3cd',
    lightBlue: '#cce5ff'
};

// Chart instances for cleanup
let charts = {
    sentiment: null,
    timeline: null,
    issues: null,
    feedback: null
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Show loading state
    showLoading();
    
    // Fetch data with retry logic
    fetchDashboardDataWithRetry(3);
});

// Function to refresh dashboard
function refreshDashboard() {
    // Show loading state
    showLoading();
    
    // Clear any existing error messages
    hideError();
    hideNoData();
    
    // Fetch data with retry logic
    fetchDashboardDataWithRetry(3);
}

async function fetchDashboardDataWithRetry(maxRetries) {
    let retries = 0;
    while (retries < maxRetries) {
        try {
            const response = await fetch('/api/dashboard/stats');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Hide loading state
            hideLoading();
            
            // Check if we have any data
            if (data.total_reviews === 0) {
                showNoData();
                return;
            }
            
            // Update dashboard with data
            updateDashboard(data);
            return;
        } catch (error) {
            retries++;
            console.error(`Attempt ${retries} failed:`, error);
            
            if (retries === maxRetries) {
                hideLoading();
                showError('Failed to load dashboard data. Please try refreshing the page.');
            } else {
                // Wait before retrying (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, Math.pow(2, retries) * 1000));
            }
        }
    }
}

function updateDashboard(data) {
    try {
        // Format numbers with proper units
        const totalReviews = data.total_reviews || 0;
        const avgSentiment = (data.average_sentiment || 0).toFixed(1);
        const responseRate = (data.response_rate || 0).toFixed(1);
        const avgResponseTime = (data.average_response_time || 0).toFixed(1);

        // Update statistics with formatted values
        document.getElementById('totalReviews').textContent = totalReviews;
        document.getElementById('avgSentiment').textContent = `${avgSentiment}%`;
        document.getElementById('responseRate').textContent = `${responseRate}%`;
        document.getElementById('averageResponseTime').textContent = `${avgResponseTime} s`;

        // Destroy existing charts if they exist
        Object.values(charts).forEach(chart => {
            if (chart) chart.destroy();
        });

        // Create new charts
        charts.sentiment = createSentimentChart(data.sentiment_distribution || {});
        charts.timeline = createTimelineChart(data.review_timeline || { dates: [], counts: [] });
        charts.issues = createIssuesChart(data.common_issues || {});
        charts.feedback = createFeedbackChart(data.feedback_distribution || {});
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showError('Error updating dashboard display');
    }
}

function createSentimentChart(data) {
    try {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'],
                datasets: [{
                    label: 'Number of Reviews',
                    data: [
                        data.very_positive || 0,
                        data.positive || 0,
                        data.neutral || 0,
                        data.negative || 0,
                        data.very_negative || 0
                    ],
                    backgroundColor: [
                        chartColors.green,
                        chartColors.lightGreen,
                        chartColors.gray,
                        chartColors.lightRed,
                        chartColors.red
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating sentiment chart:', error);
        return null;
    }
}

function createTimelineChart(data) {
    try {
        const ctx = document.getElementById('timelineChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.dates || [],
                datasets: [{
                    label: 'Reviews per Day',
                    data: data.counts || [],
                    backgroundColor: chartColors.blue,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating timeline chart:', error);
        return null;
    }
}

function createIssuesChart(data) {
    try {
        const issues = Object.entries(data);
        const ctx = document.getElementById('issuesChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: issues.map(([issue, _]) => issue.replace(/_/g, ' ')),
                datasets: [{
                    label: 'Number of Issues',
                    data: issues.map(([_, count]) => count),
                    backgroundColor: chartColors.yellow,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating issues chart:', error);
        return null;
    }
}

function createFeedbackChart(data) {
    try {
        const ctx = document.getElementById('feedbackChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Helpfulness', 'Empathy', 'Relevance', 'Clarity', 'Professionalism'],
                datasets: [{
                    label: 'Average Score',
                    data: [
                        data.helpfulness || 0,
                        data.empathy || 0,
                        data.relevance || 0,
                        data.clarity || 0,
                        data.professionalism || 0
                    ],
                    backgroundColor: chartColors.green,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 5,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating feedback chart:', error);
        return null;
    }
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loadingOverlay';
    loadingDiv.className = 'loading-overlay';
    loadingDiv.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Loading dashboard data...</p>
    `;
    document.body.appendChild(loadingDiv);
}

function hideLoading() {
    const loadingDiv = document.getElementById('loadingOverlay');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function showError(message) {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.textContent = message;
    errorContainer.style.display = 'block';
}

function hideError() {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.style.display = 'none';
}

function showNoData() {
    const noDataContainer = document.getElementById('noDataContainer');
    noDataContainer.style.display = 'block';
}

function hideNoData() {
    const noDataContainer = document.getElementById('noDataContainer');
    noDataContainer.style.display = 'none';
} 