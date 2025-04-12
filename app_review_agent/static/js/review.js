// Review processing logic 

// Global settings state
let responseSettings = {
    empathyLevel: 3,
    creativityLevel: 3,
    includePersonalExperience: false,
    includeExamples: false,
    includeMetaphors: false
};

let currentReview = null;
let currentResponse = null;

function processReview() {
    const reviewText = document.getElementById('reviewText').value.trim();
    if (!reviewText) {
        showAlert('Please enter a review text', 'warning');
        return;
    }

    // Disable button and show loading state
    const generateButton = document.getElementById('generateButton');
    const originalButtonText = generateButton.innerHTML;
    generateButton.disabled = true;
    generateButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Generating...';

    // Make API call with settings
    fetch('/process_review', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            review_text: reviewText,
            settings: responseSettings
        })
    })
    .then(response => response.json())
    .then(data => {
        updateResponseUI(data);
        showAlert('Response generated successfully!', 'success');
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error generating response', 'danger');
    })
    .finally(() => {
        generateButton.disabled = false;
        generateButton.innerHTML = originalButtonText;
    });
}

function updateSentimentSection(sentiment) {
    if (!sentiment) return;

    const label = document.getElementById('sentimentLabel');
    const score = document.getElementById('sentimentScore');
    const aspects = document.getElementById('sentimentAspects');

    // Update sentiment label with appropriate color
    label.textContent = sentiment.label;
    label.className = 'badge ' + getSentimentBadgeClass(sentiment.label);

    // Update score
    score.textContent = (sentiment.score * 100).toFixed(1) + '%';

    // Update aspects
    if (sentiment.aspects && sentiment.aspects.length > 0) {
        aspects.innerHTML = sentiment.aspects.map(aspect => 
            `<span class="badge bg-secondary me-2">${aspect}</span>`
        ).join('');
    } else {
        aspects.innerHTML = '<em>No specific aspects identified</em>';
    }
}

function updateResponseSection(response, context) {
    if (!response) return;

    // Update context
    const contextContent = document.getElementById('contextContent');
    if (context && context.length > 0) {
        contextContent.innerHTML = `<pre class="mb-0">${escapeHtml(context)}</pre>`;
    } else {
        contextContent.innerHTML = '<em>No specific context used</em>';
    }

    // Update response
    const responseContent = document.getElementById('responseContent');
    responseContent.innerHTML = `<pre class="mb-0">${escapeHtml(response)}</pre>`;
}

function clearForm() {
    // Clear the form
    document.getElementById('reviewForm').reset();

    // Hide results containers
    document.getElementById('sentimentContainer').style.display = 'none';
    document.getElementById('responseContainer').style.display = 'none';

    // Clear any alerts
    const alertContainer = document.getElementById('alertContainer');
    if (alertContainer) {
        alertContainer.innerHTML = '';
    }
}

function showAlert(message, type) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;

    const alertId = 'alert-' + Date.now();
    const alertHtml = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    alertContainer.innerHTML = alertHtml;

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alertElement = document.getElementById(alertId);
        if (alertElement) {
            alertElement.remove();
        }
    }, 5000);
}

function getSentimentBadgeClass(sentiment) {
    const classes = {
        'positive': 'bg-success',
        'negative': 'bg-danger',
        'neutral': 'bg-secondary',
        'mixed': 'bg-warning'
    };
    return classes[sentiment?.toLowerCase()] || 'bg-secondary';
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function updateResponseUI(data) {
    // Show containers
    document.getElementById('sentimentContainer').style.display = 'block';
    document.getElementById('responseContainer').style.display = 'block';

    // Update sentiment section
    const sentimentLabel = document.getElementById('sentimentLabel');
    const sentimentScore = document.getElementById('sentimentScore');
    
    if (data.sentiment) {
        sentimentLabel.textContent = data.sentiment.label || 'neutral';
        sentimentLabel.className = `badge ${getSentimentBadgeClass(data.sentiment.label)}`;
        sentimentScore.textContent = `${((data.sentiment.score || 0.5) * 100).toFixed(1)}%`;
    }

    // Update aspects section
    const aspectsContainer = document.getElementById('aspectsContainer');
    if (aspectsContainer && data.aspects && data.aspects.length > 0) {
        aspectsContainer.innerHTML = data.aspects.map(aspect => `
            <div class="aspect-item">
                <span class="badge bg-info">${aspect.category}</span>
                <p class="mb-0 mt-1">${aspect.text}</p>
            </div>
        `).join('');
        aspectsContainer.style.display = 'block';
    } else if (aspectsContainer) {
        aspectsContainer.style.display = 'none';
    }

    // Update response section
    const responseContent = document.getElementById('responseContent');
    if (responseContent) {
        if (data.response) {
            responseContent.innerHTML = data.response.split('\n').map(line => 
                line.trim() ? `<p>${line}</p>` : '<br>'
            ).join('');
        } else {
            responseContent.innerHTML = '<p>No response generated.</p>';
        }
    }

    // Update context section
    const contextContent = document.getElementById('contextContent');
    if (contextContent) {
        if (data.context) {
            contextContent.innerHTML = data.context.split('\n').map(line => 
                line.trim() ? `<p>${line}</p>` : '<br>'
            ).join('');
            document.getElementById('contextSection').style.display = 'block';
        } else {
            document.getElementById('contextSection').style.display = 'none';
        }
    }

    // Update sentiment analysis section
    if (data.sentiment && data.sentiment.analysis) {
        const analysis = data.sentiment.analysis;
        const analysisHtml = `
            <div class="sentiment-analysis">
                <div class="analysis-header d-flex justify-content-between align-items-center">
                    <h6>Detailed Analysis</h6>
                    <span class="badge ${getEmotionBadgeClass(analysis.primary_emotion)}">
                        ${analysis.primary_emotion}
                    </span>
                </div>
                
                <div class="analysis-metrics">
                    <div class="metric-item">
                        <label>Sentiment Score</label>
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${analysis.sentiment_score * 100}%"
                                 aria-valuenow="${analysis.sentiment_score * 100}" 
                                 aria-valuemin="0" aria-valuemax="100">
                                ${(analysis.sentiment_score * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-item">
                        <label>Urgency Level</label>
                        <span class="badge ${getUrgencyBadgeClass(analysis.urgency)}">
                            ${analysis.urgency}
                        </span>
                    </div>
                </div>
                
                ${analysis.key_points.length > 0 ? `
                    <div class="key-points mt-3">
                        <h6>Key Points</h6>
                        <ul class="list-unstyled">
                            ${analysis.key_points.map(point => `
                                <li><i class="bi bi-dot"></i> ${point}</li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                ${analysis.issues.length > 0 ? `
                    <div class="issues mt-3">
                        <h6>Identified Issues</h6>
                        <ul class="list-unstyled">
                            ${analysis.issues.map(issue => `
                                <li><i class="bi bi-exclamation-triangle"></i> ${issue}</li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
        
        document.getElementById('sentimentAnalysis').innerHTML = analysisHtml;
    }

    // Add animation class for smooth appearance
    document.querySelectorAll('.fade-in').forEach(el => {
        el.classList.remove('fade-in');
        void el.offsetWidth; // Trigger reflow
        el.classList.add('fade-in');
    });

    // Store current review and response
    currentReview = document.getElementById('reviewText').value;
    currentResponse = data.response;
    
    // Show feedback container
    showFeedbackContainer();
}

function showFeedbackContainer() {
    document.getElementById('feedbackContainer').style.display = 'block';
}

function submitFeedback(isGood) {
    const feedbackData = {
        review_text: currentReview,
        response: currentResponse,
        is_good: isGood,
        empathy_score: parseInt(document.getElementById('empathyScore').value),
        relevance_score: parseInt(document.getElementById('relevanceScore').value),
        helpfulness_score: parseInt(document.getElementById('helpfulnessScore').value),
        settings: responseSettings,
        comments: document.getElementById('feedbackComments').value
    };

    fetch('/submit_feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showAlert('Thank you for your feedback!', 'success');
            // Clear feedback form
            document.getElementById('feedbackComments').value = '';
            document.getElementById('empathyScore').value = 3;
            document.getElementById('relevanceScore').value = 3;
            document.getElementById('helpfulnessScore').value = 3;
        } else {
            showAlert('Failed to submit feedback', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error submitting feedback', 'danger');
    });
}

// Add rating functionality
function rateResponse(score) {
    document.getElementById('helpfulnessScore').value = score;
    // Update button styles
    document.querySelectorAll('.rating-container button').forEach(btn => {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-outline-primary');
    });
    event.target.classList.remove('btn-outline-primary');
    event.target.classList.add('btn-primary');
}

// Initialize UI controls
document.addEventListener('DOMContentLoaded', function() {
    // Initialize empathy level
    const empathySlider = document.getElementById('empathyLevel');
    const empathyBadge = document.getElementById('empathyLevelBadge');
    
    empathySlider.addEventListener('input', function() {
        updateEmpathyBadge(this.value);
    });

    // Initialize creativity level
    const creativitySlider = document.getElementById('creativityLevel');
    const creativityBadge = document.getElementById('creativityLevelBadge');
    
    creativitySlider.addEventListener('input', function() {
        updateCreativityBadge(this.value);
    });

    // Initialize checkboxes
    document.getElementById('includePersonalExperience').addEventListener('change', updateSettings);
    document.getElementById('includeExamples').addEventListener('change', updateSettings);
    document.getElementById('includeMetaphors').addEventListener('change', updateSettings);

    // Add event listener for form submission
    const form = document.getElementById('reviewForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            processReview();
        });
    }

    // Add keyboard shortcut (Ctrl/Cmd + Enter) to generate response
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            processReview();
        }
    });
});

function updateEmpathyBadge(value) {
    const badge = document.getElementById('empathyLevelBadge');
    const labels = {
        1: 'Professional',
        2: 'Formal',
        3: 'Balanced',
        4: 'Empathetic',
        5: 'Highly Empathetic'
    };
    badge.textContent = labels[value];
    responseSettings.empathyLevel = parseInt(value);
}

function updateCreativityBadge(value) {
    const badge = document.getElementById('creativityLevelBadge');
    const labels = {
        1: 'Conservative',
        2: 'Factual',
        3: 'Balanced',
        4: 'Creative',
        5: 'Highly Creative'
    };
    badge.textContent = labels[value];
    responseSettings.creativityLevel = parseInt(value);
}

function updateSettings() {
    responseSettings.includePersonalExperience = document.getElementById('includePersonalExperience').checked;
    responseSettings.includeExamples = document.getElementById('includeExamples').checked;
    responseSettings.includeMetaphors = document.getElementById('includeMetaphors').checked;
}

function regenerateResponse() {
    // Re-process the review with current settings
    processReview();
}

function copyResponse() {
    const responseContent = document.getElementById('responseContent');
    navigator.clipboard.writeText(responseContent.innerText)
        .then(() => showAlert('Response copied to clipboard!', 'success'))
        .catch(() => showAlert('Failed to copy response', 'danger'));
}

function getEmotionBadgeClass(emotion) {
    const classes = {
        'angry': 'bg-danger',
        'frustrated': 'bg-warning',
        'disappointed': 'bg-warning',
        'neutral': 'bg-secondary',
        'satisfied': 'bg-success',
        'happy': 'bg-success'
    };
    return classes[emotion] || 'bg-secondary';
}

function getUrgencyBadgeClass(urgency) {
    const classes = {
        'high': 'bg-danger',
        'medium': 'bg-warning',
        'low': 'bg-info'
    };
    return classes[urgency] || 'bg-secondary';
} 