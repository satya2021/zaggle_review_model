// Review processing logic 

// Global settings state
let responseSettings = {
    empathyLevel: 3,
    creativityLevel: 3,
    includePersonalExperience: false,
    includeExamples: false,
    includeMetaphors: false
};

let currentReview = '';
let currentResponse = '';
let currentReviewId = null;

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
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || data.message || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Initial response data:", data);
        if (data.status === 'error') {
            throw new Error(data.error || data.message);
        }
        if (data.task_id) {
            pollTaskStatus(data.task_id, generateButton, originalButtonText);
        } else {
            updateResponseUI(data);
            showAlert('Response generated successfully!', 'success');
            generateButton.disabled = false;
            generateButton.innerHTML = originalButtonText;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert(`Error generating response: ${error.message}`, 'danger');
        generateButton.disabled = false;
        generateButton.innerHTML = originalButtonText;
    });
}

function pollTaskStatus(taskId, generateButton, originalButtonText) {
    const pollInterval = setInterval(() => {
        fetch(`/task_status/${taskId}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || data.message || `HTTP error! status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log("Task status:", data);
                if (data.status === 'SUCCESS') {
                    clearInterval(pollInterval);
                    updateResponseUI(data.result);
                    showAlert('Response generated successfully!', 'success');
                    generateButton.disabled = false;
                    generateButton.innerHTML = originalButtonText;
                } else if (data.status === 'FAILURE' || data.status === 'error') {
                    clearInterval(pollInterval);
                    throw new Error(data.error || data.message || 'Task failed');
                } else if (data.status === 'PENDING') {
                    // Continue polling
                    console.log("Task still processing...");
                }
            })
            .catch(error => {
                clearInterval(pollInterval);
                console.error('Error polling task status:', error);
                showAlert(`Error generating response: ${error.message}`, 'danger');
                generateButton.disabled = false;
                generateButton.innerHTML = originalButtonText;
            });
    }, 1000); // Poll every second

    // Stop polling after 30 seconds
    setTimeout(() => {
        clearInterval(pollInterval);
        if (generateButton.disabled) {
            generateButton.disabled = false;
            generateButton.innerHTML = originalButtonText;
            showAlert('Response generation timed out. Please try again.', 'warning');
        }
    }, 30000);
}

function updateSentimentSection(sentiment) {
    if (!sentiment) return;

    const label = document.getElementById('sentimentLabel');
    const score = document.getElementById('sentimentScore');
    const aspects = document.getElementById('sentimentAspects');

    // Update sentiment label with new styling
    label.textContent = sentiment.label;
    label.className = getSentimentBadgeClass(sentiment.label);

    // Create score display with colored indicator
    const scorePercentage = (sentiment.score * 100).toFixed(1);
    const scoreColor = getScoreColor(sentiment.score);
    score.innerHTML = `
        <div class="d-flex align-items-center">
            <div class="progress flex-grow-1" style="height: 10px;">
                <div class="progress-bar" role="progressbar" 
                     style="width: ${scorePercentage}%; background-color: ${scoreColor};"
                     aria-valuenow="${scorePercentage}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                </div>
            </div>
            <span class="ms-2" style="color: ${scoreColor};">${scorePercentage}%</span>
        </div>
    `;

    // Update aspects with new styling
    if (sentiment.issues) {
        const issuesHtml = Object.entries(sentiment.issues)
            .filter(([_, hasIssue]) => hasIssue)
            .map(([issue, _]) => `
                <span class="badge sentiment-badge sentiment-negative me-2">
                    <i class="bi bi-exclamation-triangle-fill me-1"></i>
                    ${issue.charAt(0).toUpperCase() + issue.slice(1)}
                </span>
            `).join('');
        aspects.innerHTML = issuesHtml || '<em>No issues detected</em>';
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

    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alert);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 150);
    }, 5000);
}

function getSentimentBadgeClass(sentiment) {
    if (!sentiment) return 'sentiment-badge sentiment-neutral';
    
    const sentimentMap = {
        'very positive': 'sentiment-very-positive',
        'positive': 'sentiment-positive',
        'neutral': 'sentiment-neutral',
        'negative': 'sentiment-negative',
        'very negative': 'sentiment-very-negative'
    };
    
    return `sentiment-badge ${sentimentMap[sentiment.toLowerCase()] || 'sentiment-neutral'}`;
}

function getScoreColor(score) {
    // Color gradient from negative (red) to positive (green)
    if (score <= 0.2) return '#c0392b'; // Very negative
    if (score <= 0.4) return '#e74c3c'; // Negative
    if (score <= 0.6) return '#7f8c8d'; // Neutral
    if (score <= 0.8) return '#27ae60'; // Positive
    return '#2ecc71'; // Very positive
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
    console.log("Updating UI with data:", data);

    // Show containers
    document.getElementById('sentimentContainer').style.display = 'block';
    document.getElementById('responseContainer').style.display = 'block';
    document.getElementById('botAnalysisSection').style.display = 'block';
    document.getElementById('feedbackContainer').style.display = 'block';

    // Store current review and response for feedback
    currentReview = document.getElementById('reviewText').value;
    currentResponse = data.response;
    currentReviewId = data.review_id;

    // Update response content
    if (data.response) {
        const responseContent = document.getElementById('responseContent');
        responseContent.innerHTML = `${escapeHtml(data.response)}`;
    }

    // Update sentiment section if present
    if (data.sentiment) {
        updateSentimentSection(data.sentiment);
    }

    // Update bot analysis section
    if (data.bot_analysis) {
        // Update scores
        const scoresContainer = document.getElementById('botScores');
        const scores = data.bot_analysis.scores || {};
        
        scoresContainer.innerHTML = `
            <div class="score-details mb-3">
                ${Object.entries(scores).map(([key, value]) => `
                    <div class="score-item">
                        <span class="score-label">${key.replace(/_/g, ' ')}</span>
                        <div class="progress">
                            <div class="progress-bar bg-${getScoreClass(value)}" 
                                 role="progressbar" 
                                 style="width: ${value * 100}%" 
                                 aria-valuenow="${value * 100}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                                ${(value * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        // Update flags
        const flagsContainer = document.getElementById('botFlags');
        const flags = data.bot_analysis.flags || [];
        
        if (flags.length > 0) {
            flagsContainer.innerHTML = `
                <div class="alert alert-warning mb-0">
                    <h6 class="alert-heading mb-2">Potential Issues Detected:</h6>
                    <ul class="list-unstyled mb-0">
                        ${flags.map(flag => `
                            <li class="mb-1">
                                <i class="bi bi-exclamation-triangle me-2"></i>
                                ${formatFlagMessage(flag)}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        } else {
            flagsContainer.innerHTML = `
                <div class="alert alert-success mb-0">
                    <i class="bi bi-check-circle me-2"></i>
                    No issues detected in the response
                </div>
            `;
        }
    }

    // Reset feedback form
    resetFeedbackForm();
}

function getScoreClass(score) {
    if (score >= 0.7) return 'success';
    if (score >= 0.4) return 'warning';
    return 'danger';
}

function formatFlagMessage(flag) {
    const messages = {
        'high_bot_probability': 'Response appears automated',
        'excessive_template_usage': 'Heavy use of template phrases',
        'low_genuineness': 'Response lacks personalization',
        'low_uniqueness': 'Very similar to previous responses',
        'low_context_relevance': 'May not fully address the review'
    };
    return messages[flag] || flag.replace(/_/g, ' ');
}

function showFeedbackContainer() {
    const container = document.getElementById('feedbackContainer');
    if (container) {
        container.style.display = 'block';
        container.classList.add('fade-in');
        
        // Reset feedback form
        resetFeedbackForm();
    }
}

function resetFeedbackForm() {
    // Reset ratings
    document.querySelectorAll('.rating-container button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Reset sliders - using the correct IDs from the HTML
    const empathySlider = document.getElementById('empathyScore');
    if (empathySlider) empathySlider.value = 3;
    
    const relevanceSlider = document.getElementById('relevanceScore');
    if (relevanceSlider) relevanceSlider.value = 3;
    
    const helpfulnessSlider = document.getElementById('helpfulnessScore');
    if (helpfulnessSlider) helpfulnessSlider.value = 3;
    
    // Clear comments
    const commentsField = document.getElementById('feedbackComments');
    if (commentsField) commentsField.value = '';
}

function rateResponse(rating) {
    // Remove active class from all buttons
    document.querySelectorAll('.rating-container button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to clicked button
    const clickedButton = document.querySelector(`.rating-container button:nth-child(${rating})`);
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
    
    // Store rating
    document.getElementById('helpfulnessScore').value = rating;
}

function submitFeedback(isGood) {
    if (!currentReviewId) {
        showAlert('Error: No review ID found. Please try processing the review again.', 'danger');
        return;
    }

    // Get scores with null checks
    const empathyScore = document.getElementById('empathyScore')?.value || 3;
    const relevanceScore = document.getElementById('relevanceScore')?.value || 3;
    const helpfulnessScore = document.getElementById('helpfulnessScore')?.value || 3;
    
    const feedbackData = {
        review_id: currentReviewId,
        response: currentResponse,  // Add the current response
        feedback_data: {  // Wrap the feedback data in a feedback_data object
            helpfulness_score: parseInt(helpfulnessScore),
            empathy_score: parseInt(empathyScore),
            relevance_score: parseInt(relevanceScore),
            is_good: isGood
        }
    };
    
    // Show loading state on buttons
    const goodButton = document.querySelector('#feedbackContainer .btn-success');
    const badButton = document.querySelector('#feedbackContainer .btn-danger');
    const originalGoodHtml = goodButton?.innerHTML;
    const originalBadHtml = badButton?.innerHTML;
    
    if (goodButton) goodButton.disabled = true;
    if (badButton) badButton.disabled = true;
    
    if (isGood && goodButton) {
        goodButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Submitting...';
    } else if (!isGood && badButton) {
        badButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Submitting...';
    }
    
    // Submit feedback
    fetch('/submit_feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || data.message || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        showAlert('Thank you for your feedback!', 'success');
        // Hide feedback container with animation
        const feedbackContainer = document.getElementById('feedbackContainer');
        if (feedbackContainer) {
            feedbackContainer.classList.add('fade-out');
            setTimeout(() => {
                feedbackContainer.style.display = 'none';
                feedbackContainer.classList.remove('fade-out');
            }, 300);
        }
    })
    .catch(error => {
        console.error('Feedback submission error:', error);
        showAlert('Error submitting feedback: ' + error.message, 'danger');
    })
    .finally(() => {
        // Restore buttons to original state
        if (goodButton) {
            goodButton.disabled = false;
            goodButton.innerHTML = originalGoodHtml;
        }
        if (badButton) {
            badButton.disabled = false;
            badButton.innerHTML = originalBadHtml;
        }
    });
}

// Helper function to generate a unique review ID
function generateReviewId(reviewText) {
    // Create a hash of the review text to use as an ID
    let hash = 0;
    for (let i = 0; i < reviewText.length; i++) {
        const char = reviewText.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(16);
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

    // Initialize any existing sentiment badges
    document.querySelectorAll('.sentiment-badge').forEach(badge => {
        const sentiment = badge.textContent.toLowerCase().trim();
        badge.className = getSentimentBadgeClass(sentiment);
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