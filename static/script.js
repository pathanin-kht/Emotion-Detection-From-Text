const form = document.getElementById('emotion-form');
        const resultDiv = document.getElementById('result');
        const errorDiv = document.getElementById('error');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('text').value;
            resultDiv.textContent = '';
            errorDiv.textContent = '';

            if (!text.trim()) {
                errorDiv.textContent = 'Please enter some text to predict emotion.';
                return;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                const result = await response.json();
                if (result.error) {
                    errorDiv.textContent = `Error: ${result.error}`;
                } else {
                    resultDiv.textContent = `Predicted Emotion: ${result.emotion}`;
                }
            } catch (error) {
                errorDiv.textContent = 'Error communicating with the server. Please try again later.';
            }
        });