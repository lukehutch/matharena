let isSidebarCollapsed = false;


document.addEventListener('DOMContentLoaded', function() {
    const dropdown = document.getElementById('comp-dropdown');
    if (dropdown) {
        dropdown.addEventListener('change', function() {
            const selectedComp = this.value.replace(/\//g, '---');
            // Redirect to /refresh/COMP/CURRENTURL
            window.location.href = `/refresh/${selectedComp}`;
        });
    }
});

function copyToClipboard() {
    // Get the parent element of the button
    const button = event.target;
    const parentDiv = button.closest('div');

    // Get all sibling P elements
    const siblings = Array.from(parentDiv.children).filter(el => el.tagName === 'P');

    // Extract innerHTML from each sibling, one per line
    const text = siblings.map(el => el.innerHTML).join('\n');

    // Copy to clipboard
    navigator.clipboard.writeText(text).then(function() {
        // Visual feedback
        const originalText = button.textContent;
        button.textContent = '[Copied!]';
        setTimeout(function() {
            button.textContent = originalText;
        }, 1500);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('Failed to copy to clipboard');
    });
}

window.MathJax = {
    tex: {
        inlineMath: [['$', '$']]
    }
};

document.addEventListener('DOMContentLoaded', function() {
    const current = document.querySelector('.sidebar-item.current');
    if (current) {
        current.scrollIntoView({ behavior: 'auto', block: 'center' });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.response-box-details').forEach(function(element) {
        element.addEventListener('toggle', function(event) {
            if (event.target.open) {
                const idd = event.target.getAttribute('id');
                if (!event.target.hasAttribute('data-loaded')) {
                    fetch(`/modelinteraction/${idd}`)
                        .then(response => response.text())
                        .then(data => {
                            const wrapper = document.createElement('div');
                            wrapper.className = 'conversation-content';
                            wrapper.innerHTML = data;
                            event.target.appendChild(wrapper);
                            event.target.setAttribute('data-loaded', true);
                            renderMathInElement(wrapper,  {
                                delimiters: [
                                    { left: '$$', right: '$$', display: true },
                                    { left: '$', right: '$', display: false },
                                    { left: '\\(', right: '\\)', display: false },
                                    { left: '\\[', right: '\\]', display: true }
                                ]
                            });
                            hljs.highlightAll();
                        })
                        .catch(error => console.error('Error fetching details:', error));
                }
            }
        });
    });
});

function loadHistoryStep(stepId) {
    // split stepId by ">>"
    
    if (!stepId) {
        for (const element of document.getElementsByClassName('history-step-content')) {
            element.innerHTML = '';
        }
        return;
    }

    const parts = stepId.split(">>");
    const index = parts[2];

    fetch(`/historystep/${stepId}`)
        .then(response => response.text())
        .then(data => {
            document.getElementById(`history-step-content-${index}`).innerHTML = data;
            renderMathInElement(document.getElementById(`history-step-content-${index}`),  {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ]
            });
            hljs.highlightAll();
        })
        .catch(error => {
            console.error('Error fetching step:', error);
            document.getElementById(`history-step-content-${index}`).innerHTML = '<div class="error">Error loading step</div>';
        });
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.getElementById('sidebar-toggle-button');

    isSidebarCollapsed = !isSidebarCollapsed;
    localStorage.setItem('sidebarCollapsed', isSidebarCollapsed); // Persist state

    if (isSidebarCollapsed) {
        sidebar.classList.add('collapsed');
        if (toggleButton) toggleButton.innerHTML = '&#9776;'; // Hamburger icon
    } else {
        sidebar.classList.remove('collapsed');
        if (toggleButton) toggleButton.innerHTML = '&times;'; // Close icon
    }
}