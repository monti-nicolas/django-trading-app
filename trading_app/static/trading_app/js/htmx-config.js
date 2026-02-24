/**
 * HTMX configuration and custom behaviors
 */

// Configure HTMX defaults
document.addEventListener('DOMContentLoaded', function() {
    // Set default HTMX timeout
    if (typeof htmx !== 'undefined') {
        htmx.config.timeout = 30000; // 30 seconds
        htmx.config.useTemplateFragments = true;

        console.log('HTMX configured');
    }
});

// Add loading indicator during HTMX requests
document.body.addEventListener('htmx:beforeRequest', function(event) {
    const target = event.detail.target;
    if (target) {
        // Add loading class
        target.classList.add('loading');

        // Add loading spinner if target is a container
        if (target.id === 'ticker-details-container') {
            target.innerHTML = `
                <div class="text-center p-5">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-3">Loading ticker details...</p>
                </div>
            `;
        }
    }
});

// Remove loading indicator after HTMX request
document.body.addEventListener('htmx:afterRequest', function(event) {
    const target = event.detail.target;
    if (target) {
        target.classList.remove('loading');
    }
});

// Handle HTMX errors
document.body.addEventListener('htmx:responseError', function(event) {
    console.error('HTMX Response Error:', event.detail);

    const target = event.detail.target;
    if (target) {
        target.innerHTML = `
            <div class="alert alert-danger-custom m-3" role="alert">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error:</strong> Failed to load data. Please try again.
            </div>
        `;
    }
});

// Handle HTMX network errors
document.body.addEventListener('htmx:sendError', function(event) {
    console.error('HTMX Send Error:', event.detail);

    alert('Network error. Please check your connection and try again.');
});

// Log HTMX events in development
document.body.addEventListener('htmx:afterSwap', function(event) {
    console.log('HTMX swapped content for:', event.detail.target.id);
});

// Prevent multiple simultaneous requests to same endpoint
let pendingRequests = new Set();

document.body.addEventListener('htmx:beforeRequest', function(event) {
    const url = event.detail.pathInfo.requestPath;

    if (pendingRequests.has(url)) {
        console.warn('Request already pending for:', url);
        event.preventDefault();
        return;
    }

    pendingRequests.add(url);
});

document.body.addEventListener('htmx:afterRequest', function(event) {
    const url = event.detail.pathInfo.requestPath;
    pendingRequests.delete(url);
});

// Add CSRF token to HTMX requests
document.body.addEventListener('htmx:configRequest', function(event) {
    const csrftoken = getCookie('csrftoken');
    if (csrftoken) {
        event.detail.headers['X-CSRFToken'] = csrftoken;
    }
});

// Helper function to get cookie (shared with dashboard.js)
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}