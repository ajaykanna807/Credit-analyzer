document.getElementById('creditForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const formData = {
    credit_score: +document.getElementById('creditScore').value,
    income: +document.getElementById('income').value,
    spending: +document.getElementById('spending').value,
    social_media_sentiment: +document.getElementById('socialMediaSentiment').value,
    geolocation_stability: +document.getElementById('geoStability').value,
    utility_bill_reliability: +document.getElementById('utilityReliability').value,
    purchase_to_income_ratio: +document.getElementById('purchaseToIncomeRatio').value
  };

  const outputDiv = document.getElementById('output');
  outputDiv.textContent = 'Processing...';

  try {
    const response = await fetch('http://127.0.0.1:5000/assess_credit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });

    const result = await response.json();
    if (response.ok) {
      outputDiv.textContent = `Risk: ${result.credit_risk}`;
    } else {
      outputDiv.textContent = `Error: ${result.error}`;
    }
  } catch (error) {
    outputDiv.textContent = 'Failed to connect to the server.';
  }
});
