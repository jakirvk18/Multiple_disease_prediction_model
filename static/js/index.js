        document.addEventListener('DOMContentLoaded', function() {
            const navLinks = document.querySelectorAll('.nav-link');
            const predictionFrame = document.getElementById('prediction-frame');
            const initialMessage = document.getElementById('initial-message');

            // Function to remove 'active-nav-link' class from all links
            function removeActiveClass() {
                navLinks.forEach(link => {
                    link.classList.remove('active-nav-link');
                });
            }

            // Function to set initial state: blank frame and show message
            function setInitialState() {
                predictionFrame.src = ""; // Ensure iframe is blank
                predictionFrame.style.display = 'none'; // Hide the iframe
                initialMessage.style.display = 'block'; // Show the initial message
                removeActiveClass(); // No nav link is active initially
            }

            // Set initial state on page load
            setInitialState();
            
            // Add click event listeners to navigation links
            navLinks.forEach(link => {
                link.addEventListener('click', function(event) {
                    event.preventDefault(); // Prevent default link behavior (page reload)

                    removeActiveClass(); // Remove active class from all links
                    this.classList.add('active-nav-link'); // Add active class to the clicked link

                    // Determine which content to load based on the link's ID
                    let contentUrl = '';
                    if (this.id === 'diabetes-nav-link') {
                        contentUrl = '/get_prediction_content/diabetes';
                    } else if (this.id === 'heart-nav-link') {
                        contentUrl = '/get_prediction_content/heart_disease';
                    } else if (this.id === 'lung-nav-link') {
                        contentUrl = '/get_prediction_content/lung_disease';
                    } else if (this.id === 'home-nav-link') {
                        // If 'Home' is clicked, revert to initial blank state
                        setInitialState();
                        return; // Exit function, no need to load iframe
                    }

                    // Load content into the iframe
                    if (contentUrl) {
                        predictionFrame.src = contentUrl;
                        predictionFrame.style.display = 'block'; // Show the iframe
                        initialMessage.style.display = 'none'; // Hide the initial message
                    } else {
                        console.warn('No content URL defined for this link:', this.id);
                        setInitialState(); // Fallback to initial state if no URL
                    }
                });
            });
        });
