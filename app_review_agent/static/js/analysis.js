// Analysis display logic 

// Chart color scheme
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

// Initialize all charts
function initializeCharts() {
    fetchAnalyticsData().then(data => {
        updateMetrics(data);
        createSentimentChart(data.sentiment_distribution);
        createIssuesChart(data.common_issues);
        createTimelineChart(data.review_timeline);
        createFeedbackChart(data.feedback_distribution);
        updateDetailedAnalysis(data.detailed_analysis);
    });
}

// Update summary metrics
function updateMetrics(data) {
    document.getElementById('totalReviews').textContent = data.total_reviews;
    document.getElementById('avgSentiment').textContent = `${data.average_sentiment}%`;
    document.getElementById('responseRate').textContent = `${data.response_rate}%`;
    document.getElementById('avgResponseTime').textContent = `${data.average_response_time}s`;
}

// Create sentiment distribution chart
function createSentimentChart(data) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'],
            datasets: [{
                data: [
                    data.very_positive,
                    data.positive,
                    data.neutral,
                    data.negative,
                    data.very_negative
                ],
                backgroundColor: [
                    chartColors.green,
                    chartColors.lightGreen,
                    chartColors.gray,
                    chartColors.lightRed,
                    chartColors.red
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

// Create common issues chart
function createIssuesChart(data) {
    const ctx = document.getElementById('issuesChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Number of Issues',
                data: Object.values(data),
                backgroundColor: chartColors.lightRed,
                borderColor: chartColors.red,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Create timeline chart
function createTimelineChart(data) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: 'Number of Reviews',
                data: data.counts,
                borderColor: chartColors.blue,
                backgroundColor: chartColors.lightBlue,
                fill: true
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Create feedback distribution chart
function createFeedbackChart(data) {
    const ctx = document.getElementById('feedbackChart').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Helpfulness', 'Empathy', 'Relevance', 'Clarity', 'Professionalism'],
            datasets: [{
                label: 'Average Ratings',
                data: [
                    data.helpfulness,
                    data.empathy,
                    data.relevance,
                    data.clarity,
                    data.professionalism
                ],
                backgroundColor: chartColors.lightBlue,
                borderColor: chartColors.blue,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 5
                }
            }
        }
    });
}

// Fetch analytics data from the server
async function fetchAnalyticsData() {
    try {
        const response = await fetch('/api/analysis/stats');
        if (!response.ok) {
            throw new Error('Failed to fetch analytics data');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching analytics data:', error);
        showError('Failed to load analytics data. Please try again later.');
        return {
            total_reviews: 0,
            average_sentiment: 0,
            response_rate: 0,
            average_response_time: 0,
            sentiment_distribution: {
                very_positive: 0,
                positive: 0,
                neutral: 0,
                negative: 0,
                very_negative: 0
            },
            common_issues: {},
            review_timeline: {
                dates: [],
                counts: []
            },
            feedback_distribution: {
                helpfulness: 0,
                empathy: 0,
                relevance: 0,
                clarity: 0,
                professionalism: 0
            }
        };
    }
}

// Show error message
function showError(message) {
    const alertContainer = document.getElementById('alertContainer');
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    alertContainer.appendChild(alert);
    setTimeout(() => alert.remove(), 5000);
}

// Initialize charts when the page loads
document.addEventListener('DOMContentLoaded', initializeCharts);

function updateDetailedAnalysis(data) {
    if (!data) return;
    
    // Update Recent Reviews
    const recentReviewsContainer = document.getElementById('recentReviews');
    if (recentReviewsContainer && data.recent_reviews) {
        // Filter out reviews with no text and ensure we have valid data
        const validReviews = data.recent_reviews
            .filter(review => review && review.text && review.text.trim() !== '')
            .map(review => ({
                review_date: review.date || new Date().toISOString(),
                rating: review.rating || 0,
                review_text: review.text || '',
                sentiment: review.sentiment?.label || 'Neutral'
            }));
        
        if (validReviews.length > 0) {
            recentReviewsContainer.innerHTML = `
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Rating</th>
                                <th>Review</th>
                                <th>Sentiment</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${validReviews.map(review => `
                                <tr>
                                    <td>${new Date(review.review_date).toLocaleDateString()}</td>
                                    <td>
                                        <span class="badge bg-${getRatingBadgeClass(review.rating)}">
                                            ${review.rating} ★
                                        </span>
                                    </td>
                                    <td>${review.review_text}</td>
                                    <td>
                                        <span class="badge ${getSentimentBadgeClass(review.sentiment)}">
                                            ${review.sentiment}
                                        </span>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        } else {
            recentReviewsContainer.innerHTML = '<p class="text-muted">No recent reviews available.</p>';
        }
    }

    // Update Top Issues
    const topIssuesContainer = document.getElementById('topIssues');
    if (topIssuesContainer && data.top_issues) {
        topIssuesContainer.innerHTML = `
            <div class="list-group">
                ${data.top_issues.map(issue => `
                    <div class="list-group-item">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-1">${issue.issue}</h6>
                            <span class="badge bg-primary rounded-pill">${issue.count}</span>
                        </div>
                        <p class="mb-1 text-muted">${issue.description || 'No description available'}</p>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Update Response Quality
    const responseQualityContainer = document.getElementById('responseQuality');
    if (responseQualityContainer && data.response_quality) {
        responseQualityContainer.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">Empathy</h6>
                            <div class="progress">
                                <div class="progress-bar bg-success" role="progressbar" 
                                     style="width: ${data.response_quality.empathy}%">
                                    ${data.response_quality.empathy}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">Relevance</h6>
                            <div class="progress">
                                <div class="progress-bar bg-info" role="progressbar" 
                                     style="width: ${data.response_quality.relevance}%">
                                    ${data.response_quality.relevance}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">Helpfulness</h6>
                            <div class="progress">
                                <div class="progress-bar bg-warning" role="progressbar" 
                                     style="width: ${data.response_quality.helpfulness}%">
                                    ${data.response_quality.helpfulness}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Update Sentiment Trends
    const sentimentTrendsContainer = document.getElementById('sentimentTrends');
    if (sentimentTrendsContainer && data.sentiment_trends) {
        sentimentTrendsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">Last Week</h6>
                            <div class="d-flex align-items-center">
                                <span class="badge ${getTrendColor(data.sentiment_trends.last_week)} me-2">
                                    ${data.sentiment_trends.last_week}%
                                </span>
                                <small class="text-muted">vs previous week</small>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">Last Month</h6>
                            <div class="d-flex align-items-center">
                                <span class="badge ${getTrendColor(data.sentiment_trends.last_month)} me-2">
                                    ${data.sentiment_trends.last_month}%
                                </span>
                                <small class="text-muted">vs previous month</small>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">Last Quarter</h6>
                            <div class="d-flex align-items-center">
                                <span class="badge ${getTrendColor(data.sentiment_trends.last_quarter)} me-2">
                                    ${data.sentiment_trends.last_quarter}%
                                </span>
                                <small class="text-muted">vs previous quarter</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

// Helper function to get badge class based on rating
function getRatingBadgeClass(rating) {
    if (rating >= 4) return 'success';
    if (rating >= 3) return 'warning';
    return 'danger';
}

// Helper function to get badge class based on sentiment
function getSentimentBadgeClass(sentiment) {
    switch(sentiment.toLowerCase()) {
        case 'positive': return 'bg-success';
        case 'negative': return 'bg-danger';
        case 'neutral': return 'bg-secondary';
        default: return 'bg-secondary';
    }
}

// Helper function to get trend color
function getTrendColor(value) {
    if (value > 0) return 'bg-success';
    if (value < 0) return 'bg-danger';
    return 'bg-secondary';
} 