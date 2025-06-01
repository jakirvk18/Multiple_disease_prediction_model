    document.addEventListener('DOMContentLoaded', function() {
        const form = document.querySelector('form');
        const predictionResultDiv = document.getElementById('prediction-result');
        let currentPredictionData = null; // Store the last prediction data for LLM insight

        form.addEventListener('submit', function(event) {
            console.log('Lung Cancer Form submitted! Preventing default behavior...');
            event.preventDefault(); // Prevent the default form submission (page reload)

            // Clear previous results and show loading message
            predictionResultDiv.innerHTML = '<p>Predicting risk... Please wait.</p>';
            predictionResultDiv.style.backgroundColor = '#f0f0f0';
            predictionResultDiv.style.color = '#333';
            predictionResultDiv.style.border = '1px solid #ccc';
            predictionResultDiv.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.05)';

            // Collect form data
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });
            currentPredictionData = data; // Store raw input data for LLM prompt

            console.log('Sending data to backend:', data);

            // Send data to the Flask backend using fetch
            fetch('/predict/lung', { // This URL must match the Flask route for lung prediction
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                console.log('Received response status:', response.status);
                if (!response.ok) {
                    return response.json().then(errorData => {
                        console.error('Backend error response:', errorData);
                        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Prediction data received:', data);
                const riskPercentage = data.risk_percentage; // Get the risk percentage
                const predictionStatus = data.prediction; // Get the 0/1 prediction status for tips

                // Store the full prediction data for potential LLM use
                currentPredictionData = { ...currentPredictionData, prediction: predictionStatus, risk_percentage: riskPercentage };

                // Determine prediction text (optional, but good for clarity)
                let predictionText = predictionStatus === 1 ? "High Risk" : "Low Risk";
                let borderColor = predictionStatus === 1 ? '#cc0000' : '#008000'; // Red for high, green for low

                // Format tips into an unordered list if tips exist
                let tipsHtml = data.tips && data.tips.length > 0
                               ? `<h4>Health Tips:</h4><ul>${data.tips.map(tip => `<li>${tip}</li>`).join('')}</ul>`
                               : '';

                // --- START OF ANIMATION-RELATED CHANGES ---

                // 1. First, render the HTML with the bar's width set to 0%
                predictionResultDiv.innerHTML = `
                    <h3>Lung Cancer Risk Prediction</h3>
                    <div class="risk-chart-container">
                        <div class="risk-bar-label">Risk Level: <span id="risk-percentage-value">0%</span></div>
                        <div class="risk-bar-background">
                            <div class="risk-bar-fill" style="width: 0%;"></div>
                        </div>
                    </div>
                    ${tipsHtml}
                    <div id="llm-insight-section">
                        <button id="llm-insight-button">✨ Get More Insights</button>
                        <div id="llm-insight-result"></div>
                    </div>
                `;
                predictionResultDiv.style.backgroundColor = 'white';
                predictionResultDiv.style.color = '#333';
                predictionResultDiv.style.border = `1px solid ${borderColor}`;
                predictionResultDiv.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.1)';

                // 2. Use a small timeout to allow the browser to render the 0% state
                //    before applying the final width, triggering the CSS transition.
                setTimeout(() => {
                    const riskBarFill = predictionResultDiv.querySelector('.risk-bar-fill');
                    const riskPercentageValueSpan = predictionResultDiv.querySelector('#risk-percentage-value');

                    if (riskBarFill) {
                        riskBarFill.style.width = riskPercentage + '%';
                        if (riskPercentageValueSpan) {
                            riskPercentageValueSpan.textContent = riskPercentage + '%';
                        }
                    }
                }, 500); // A small delay (e.g., 50 milliseconds)

                // --- END OF ANIMATION-RELATED CHANGES ---

                // --- LLM Insight Button Event Listener ---
                const llmInsightButton = predictionResultDiv.querySelector('#llm-insight-button');
                const llmInsightResultDiv = predictionResultDiv.querySelector('#llm-insight-result');

                if (llmInsightButton) {
                    llmInsightButton.addEventListener('click', function() {
                        llmInsightResultDiv.innerHTML = '<p>Generating insights... Please wait. <i class="fa-solid fa-spinner fa-spin"></i></p>';
                        llmInsightResultDiv.style.backgroundColor = '#e6f7ff'; // Light blue for loading
                        llmInsightResultDiv.style.color = '#333';

                        fetch('/get_llm_insight', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                disease_type: 'lung', // Indicate disease type
                                prediction_data: currentPredictionData // Send all relevant data
                            })
                        })
                        .then(response => {
                            if (!response.ok) {
                                return response.json().then(errorData => {
                                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                                });
                            }
                            return response.json();
                        })
                        .then(llmData => {
                            llmInsightResultDiv.innerHTML = `<p>${llmData.insight}</p>`;
                            llmInsightResultDiv.style.backgroundColor = '#f9f9f9'; // Reset background
                            llmInsightResultDiv.style.color = '#444';
                        })
                        .catch(error => {
                            console.error('Error fetching LLM insight:', error);
                            llmInsightResultDiv.innerHTML = `<p style="color: red;">Error generating insights: ${error.message}</p>`;
                            llmInsightResultDiv.style.backgroundColor = '#ffe0e0';
                            llmInsightResultDiv.style.color = '#cc0000';
                        });
                    });
                }


            })
            .catch(error => {
                console.error('Error during fetch operation:', error);
                predictionResultDiv.style.backgroundColor = '#ffcccc';
                predictionResultDiv.style.color = '#cc0000';
                predictionResultDiv.style.border = '1px solid #cc0000';
                predictionResultDiv.innerHTML = `<h3>Error:</h3><p>${error.message || 'An unknown error occurred. Please try again.'}</p>`;
                predictionResultDiv.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.1)';
            });
        });

        const resetButton = form.querySelector('button[type="reset"]');
        if (resetButton) {
            resetButton.addEventListener('click', function() {
                predictionResultDiv.innerHTML = '';
                predictionResultDiv.style.backgroundColor = 'transparent';
                predictionResultDiv.style.color = '#333';
                predictionResultDiv.style.border = 'none';
                predictionResultDiv.style.boxShadow = 'none';
            });
        }
    });