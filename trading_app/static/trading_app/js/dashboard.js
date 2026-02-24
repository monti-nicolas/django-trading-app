// Global state
let isUpdating = false;
let selectedTicker = null;
let updateModal = null;

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');

    // Initialize Bootstrap modal
    const modalElement = document.getElementById('updateModal');
    if (modalElement) {
        updateModal = new bootstrap.Modal(modalElement);
    }

    // Get initial ticker from window data
    if (window.dashboardData && window.dashboardData.selectedTicker) {
        selectedTicker = window.dashboardData.selectedTicker;
        loadAllCharts(selectedTicker);
    }

    // Setup HTMX event listeners
    setupHTMXListeners();
});

/**
 * Update data by calling the backend
 */
function updateData() {
    if (isUpdating) {
        alert('Update already in progress. Please wait...');
        return;
    }

    isUpdating = true;

    // Show modal
    if (updateModal) {
        updateModal.show();
    }

    // Update status text
    updateStatusText('Fetching latest market data...');

    // Get CSRF token
    const csrftoken = getCookie('csrftoken');

    // Make AJAX request to update endpoint
    fetch('/update/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateStatusText(`Update completed in ${data.duration}s. Processed ${data.tickers_processed} tickers.`);

            // Wait 2 seconds then reload page
            setTimeout(() => {
                if (updateModal) {
                    updateModal.hide();
                }
                location.reload();
            }, 2000);
        } else {
            updateStatusText(`Error: ${data.message || data.error}`);
            setTimeout(() => {
                if (updateModal) {
                    updateModal.hide();
                }
                isUpdating = false;
            }, 3000);
        }
    })
    .catch(error => {
        console.error('Error updating data:', error);
        updateStatusText('Error: Failed to update data. Please check your connection.');
        setTimeout(() => {
            if (updateModal) {
                updateModal.hide();
            }
            isUpdating = false;
        }, 3000);
    });
}

/**
 * Update the status text in the modal
 */
function updateStatusText(text) {
    const statusElement = document.getElementById('update-status-text');
    if (statusElement) {
        statusElement.textContent = text;
    }
}

/**
 * Select a ticker from the sidebar
 */
function selectTicker(ticker, event) {
    if (event) {
        event.preventDefault();
    }

    // Update selected ticker
    selectedTicker = ticker;

    // Update active state in sidebar
    document.querySelectorAll('.ticker-item').forEach(item => {
        item.classList.remove('active');
    });

    const clickedItem = event ? event.currentTarget : document.querySelector(`[data-ticker="${ticker}"]`);
    if (clickedItem) {
        clickedItem.classList.add('active');
    }

    // HTMX will handle the content update
    // Charts will be reloaded by the ticker_details.html partial
}

/**
 * Setup HTMX event listeners
 */
function setupHTMXListeners() {
    // Listen for HTMX after swap events
    document.body.addEventListener('htmx:afterSwap', function(event) {
        console.log('HTMX content swapped');

        // If ticker details were swapped, reload charts
        if (event.detail.target.id === 'ticker-details-container') {
            if (selectedTicker) {
                loadAllCharts(selectedTicker);
            }
        }
    });

    // Listen for HTMX before request
    document.body.addEventListener('htmx:beforeRequest', function(event) {
        console.log('HTMX request starting');
    });

    // Listen for HTMX errors
    document.body.addEventListener('htmx:responseError', function(event) {
        console.error('HTMX error:', event.detail);
        alert('Error loading data. Please try again.');
    });
}

/**
 * Get CSRF token from cookies
 */
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

/**
 * Format number as currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

/**
 * Format number with commas
 */
function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

/**
 * Show loading spinner in a container
 */
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="chart-loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }
}

/**
 * Show error message in a container
 */
function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger-custom" role="alert">
                <i class="bi bi-exclamation-triangle"></i> ${message}
            </div>
        `;
    }
}

/**
 * Toggle sidebar collapse/expand
 */
function toggleSidebar() {
    console.log('toggleSidebar called');

    const sidebar = document.getElementById('sidebar');
    const sidebarWrapper = document.getElementById('sidebar-wrapper');
    const mainContent = document.getElementById('main-content');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');

    console.log('Elements found:', {
        sidebar: sidebar,
        sidebarWrapper: sidebarWrapper,
        mainContent: mainContent,
        toggleIcon: toggleIcon
    });

    // Toggle collapsed class
    sidebar.classList.toggle('sidebar-collapsed');
    sidebarWrapper.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');

    // Change icon direction
    if (sidebar.classList.contains('sidebar-collapsed')) {
        toggleIcon.classList.remove('bi-chevron-left');
        toggleIcon.classList.add('bi-chevron-right');
        // Save state to localStorage
        localStorage.setItem('sidebarCollapsed', 'true');
    } else {
        toggleIcon.classList.remove('bi-chevron-right');
        toggleIcon.classList.add('bi-chevron-left');
        // Save state to localStorage
        localStorage.setItem('sidebarCollapsed', 'false');
    }
}

/**
 * Restore sidebar state from localStorage on page load
 */
function restoreSidebarState() {
    const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

    if (isCollapsed) {
        const sidebar = document.getElementById('sidebar');
        const sidebarWrapper = document.getElementById('sidebar-wrapper');
        const mainContent = document.getElementById('main-content');
        const toggleIcon = document.getElementById('sidebar-toggle-icon');

        sidebar.classList.add('sidebar-collapsed');
        sidebarWrapper.classList.add('collapsed');
        mainContent.classList.add('expanded');
        toggleIcon.classList.remove('bi-chevron-left');
        toggleIcon.classList.add('bi-chevron-right');
    }
}

// Update the DOMContentLoaded listener to restore sidebar state
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');

    // Initialize Bootstrap modal
    const modalElement = document.getElementById('updateModal');
    if (modalElement) {
        updateModal = new bootstrap.Modal(modalElement);
    }

    // Get initial ticker from window data
    if (window.dashboardData && window.dashboardData.selectedTicker) {
        selectedTicker = window.dashboardData.selectedTicker;
        loadAllCharts(selectedTicker);
    }

    // Setup HTMX event listeners
    setupHTMXListeners();

    // Restore sidebar state
    restoreSidebarState();
});

/**
 * Filter tickers based on search input
 */
function filterTickers() {
    const searchInput = document.getElementById('ticker-search');
    const searchTerm = searchInput.value.toUpperCase();
    const tickerList = document.getElementById('ticker-list');
    const tickerItems = tickerList.getElementsByClassName('ticker-item');
    const clearBtn = document.getElementById('clear-search');
    const tickerCount = document.getElementById('ticker-count');

    let visibleCount = 0;

    // Show/hide clear button
    if (searchTerm.length > 0) {
        clearBtn.style.display = 'block';
    } else {
        clearBtn.style.display = 'none';
    }

    // Filter ticker items
    for (let i = 0; i < tickerItems.length; i++) {
        const ticker = tickerItems[i].getAttribute('data-ticker');

        if (ticker.toUpperCase().indexOf(searchTerm) > -1) {
            tickerItems[i].style.display = '';
            visibleCount++;
        } else {
            tickerItems[i].style.display = 'none';
        }
    }

    // Update counter
    tickerCount.textContent = `${visibleCount} of ${tickerItems.length} tickers`;
}

/**
 * Clear ticker search
 */
function clearTickerSearch() {
    const searchInput = document.getElementById('ticker-search');
    searchInput.value = '';
    filterTickers();
    searchInput.focus();
}

/**
 * Keyboard shortcuts for search
 */
document.addEventListener('keydown', function(e) {
    // Focus search on "/" key (like GitHub)
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        const searchInput = document.getElementById('ticker-search');
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Clear search on Escape
    if (e.key === 'Escape') {
        const searchInput = document.getElementById('ticker-search');
        if (searchInput && document.activeElement === searchInput) {
            clearTickerSearch();
        }
    }
});